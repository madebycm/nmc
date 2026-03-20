"""
Build reference embeddings for kNN classification.
Memory-efficient: processes crops in batches, never loads all into RAM.
"""
import json
import numpy as np
import torch
import timm
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / "data"
COCO_ROOT = DATA_ROOT / "coco_dataset" / "train"
PRODUCT_ROOT = DATA_ROOT / "product_images"
OUTPUT_DIR = SCRIPT_DIR / "submission"

MODEL_NAME = "eva02_base_patch16_clip_224"
EMBED_DIM = 768
IMG_SIZE = 224
BATCH_SIZE = 64
MAX_PER_CAT = 20


def get_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    return model


def get_transform():
    data_cfg = timm.data.resolve_data_config({}, model=MODEL_NAME)
    return timm.data.create_transform(**data_cfg, is_training=False)


@torch.no_grad()
def embed_batch(model, tensors, device):
    """Embed a batch of tensors."""
    batch = torch.stack(tensors).to(device)
    embs = model(batch).cpu().numpy()
    return embs


def stream_training_crops(transform):
    """Yield (tensor, label) pairs from training crops, grouped by image."""
    with open(COCO_ROOT / "annotations.json") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}

    # Group annotations by image
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Track per-category counts
    cat_counts = {}
    total = 0

    for img_id, anns in sorted(anns_by_img.items()):
        img_info = images[img_id]
        img_path = COCO_ROOT / "images" / img_info["file_name"]
        try:
            img = Image.open(str(img_path)).convert("RGB")
        except Exception:
            continue
        iw, ih = img.size

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_counts.get(cat_id, 0) >= MAX_PER_CAT:
                continue

            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))
            tensor = transform(crop)
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
            total += 1
            yield tensor, cat_id

    print(f"Training crops: {total} across {len(cat_counts)} categories", flush=True)


def stream_product_images(transform):
    """Yield (tensor, label) pairs from product reference images."""
    with open(COCO_ROOT / "annotations.json") as f:
        coco = json.load(f)
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}

    with open(PRODUCT_ROOT / "metadata.json") as f:
        meta = json.load(f)

    code_to_cat = {}
    for prod in meta["products"]:
        name = prod["product_name"]
        if name in cat_name_to_id:
            code_to_cat[prod["product_code"]] = cat_name_to_id[name]

    print(f"Product codes mapped: {len(code_to_cat)}", flush=True)
    total = 0

    for folder in sorted(PRODUCT_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        code = folder.name
        if code not in code_to_cat:
            continue
        cat_id = code_to_cat[code]
        for img_path in sorted(folder.glob("*.jpg")):
            try:
                img = Image.open(str(img_path)).convert("RGB")
                tensor = transform(img)
                total += 1
                yield tensor, cat_id
            except Exception:
                continue

    print(f"Product images: {total}", flush=True)


def compute_all_embeddings(model, generators, device):
    """Compute embeddings from generators in batches."""
    all_embeddings = []
    all_labels = []
    batch_tensors = []
    batch_labels = []
    processed = 0

    for gen in generators:
        for tensor, label in gen:
            batch_tensors.append(tensor)
            batch_labels.append(label)

            if len(batch_tensors) >= BATCH_SIZE:
                embs = embed_batch(model, batch_tensors, device)
                all_embeddings.append(embs)
                all_labels.extend(batch_labels)
                processed += len(batch_tensors)
                if processed % 500 == 0:
                    print(f"  Embedded {processed} crops...", flush=True)
                batch_tensors = []
                batch_labels = []

    # Final partial batch
    if batch_tensors:
        embs = embed_batch(model, batch_tensors, device)
        all_embeddings.append(embs)
        all_labels.extend(batch_labels)
        processed += len(batch_tensors)

    print(f"Total embedded: {processed}", flush=True)
    return np.concatenate(all_embeddings, axis=0), all_labels


def export_onnx(model, output_path):
    """Export timm model to ONNX, then convert to FP16."""
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embeddings": {0: "batch"},
        },
    )
    fp32_size = output_path.stat().st_size / 1024 / 1024
    print(f"FP32 ONNX: {fp32_size:.1f} MB", flush=True)

    # Convert to FP16 to fit within 420MB limit
    try:
        import onnx
        from onnxconverter_common import float16
        model_fp32 = onnx.load(str(output_path))
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        # Overwrite FP32 with FP16 to save space
        onnx.save(model_fp16, str(output_path))
        fp16_size = output_path.stat().st_size / 1024 / 1024
        print(f"FP16 ONNX (overwritten): {fp16_size:.1f} MB", flush=True)
    except ImportError:
        print("WARNING: onnxconverter_common not installed, keeping FP32", flush=True)
        print("This may exceed the 420MB submission limit!", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}", flush=True)
    model = get_model()
    transform = get_transform()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Computing embeddings on {device}...", flush=True)

    # Stream crops and compute embeddings in batches
    generators = [
        stream_training_crops(transform),
        stream_product_images(transform),
    ]
    embeddings, labels = compute_all_embeddings(model, generators, device)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    # Save
    np.save(OUTPUT_DIR / "ref_embeddings.npy", embeddings.astype(np.float16))
    with open(OUTPUT_DIR / "ref_labels.json", "w") as f:
        json.dump(labels, f)

    print(f"\nSaved ref_embeddings.npy: {embeddings.shape}", flush=True)
    print(f"Saved ref_labels.json: {len(labels)} labels", flush=True)
    print(f"Categories covered: {len(set(labels))}", flush=True)

    # Export model to ONNX (FP16)
    model = model.cpu()
    print("\nExporting classifier to ONNX (FP16)...", flush=True)
    export_onnx(model, OUTPUT_DIR / "classifier.onnx")

    # Save transform params
    data_cfg = timm.data.resolve_data_config({}, model=MODEL_NAME)
    with open(OUTPUT_DIR / "transform_config.json", "w") as f:
        json.dump({
            "input_size": list(data_cfg["input_size"]),
            "mean": list(data_cfg["mean"]),
            "std": list(data_cfg["std"]),
            "interpolation": data_cfg.get("interpolation", "bilinear"),
        }, f)
    print("Saved transform_config.json", flush=True)

    print("\n=== DONE ===", flush=True)
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.is_file():
            print(f"  {p.name}: {p.stat().st_size / 1024 / 1024:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
