"""Rebuild reference embeddings using the fine-tuned classifier's backbone."""
import json
import numpy as np
import torch
import timm
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file

COCO_ROOT = Path("/mnt/SFS-qZE4t9Aw/data/coco_dataset/train")
PRODUCT_ROOT = Path("/mnt/SFS-qZE4t9Aw/data/product_images")
CLASSIFIER_PATH = Path("/clade/ng/submission/classifier.safetensors")
DEVICE = "cuda"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
BATCH_SIZE = 128
MAX_PER_CAT = 20


def load_model():
    """Load fine-tuned EVA-02 and create a feature extractor version."""
    model = timm.create_model("eva02_base_patch16_clip_224", pretrained=False, num_classes=356)
    sd = load_file(str(CLASSIFIER_PATH))
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.train(False)
    return model.to(DEVICE)


def prepare_crop(crop):
    w, h = crop.size
    scale = 256 / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = crop.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (t - mean) / std


@torch.no_grad()
def extract_features(model, tensors):
    """Extract penultimate features (before classification head)."""
    batch = torch.stack(tensors).to(DEVICE)
    # forward_features gives us the representation before the head
    features = model.forward_features(batch)
    # For ViT models, forward_features returns [B, num_tokens, dim]
    # We need the CLS token or global average
    if features.ndim == 3:
        # Use CLS token (first token)
        features = features[:, 0]
    # L2 normalize
    features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    return features.cpu().numpy()


def stream_training_crops():
    with open(COCO_ROOT / "annotations.json") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
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
            tensor = prepare_crop(crop)
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
            total += 1
            yield tensor, cat_id
    print(f"Training crops: {total} across {len(cat_counts)} categories")


def stream_product_images():
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
                tensor = prepare_crop(img)
                total += 1
                yield tensor, cat_id
            except Exception:
                continue
    print(f"Product images: {total}")


def main():
    model = load_model()
    all_embeddings = []
    all_labels = []
    batch_tensors = []
    batch_labels = []

    for gen in [stream_training_crops(), stream_product_images()]:
        for tensor, label in gen:
            batch_tensors.append(tensor)
            batch_labels.append(label)
            if len(batch_tensors) >= BATCH_SIZE:
                embs = extract_features(model, batch_tensors)
                all_embeddings.append(embs)
                all_labels.extend(batch_labels)
                if len(all_labels) % 1000 < BATCH_SIZE:
                    print(f"  Embedded {len(all_labels)}...")
                batch_tensors = []
                batch_labels = []

    if batch_tensors:
        embs = extract_features(model, batch_tensors)
        all_embeddings.append(embs)
        all_labels.extend(batch_labels)

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\nTotal: {embeddings.shape[0]} embeddings, {len(set(all_labels))} categories")
    print(f"Embedding dim: {embeddings.shape[1]}")

    # Save
    out_emb = Path("/clade/ng/submission/ref_embeddings_finetuned.npy")
    np.save(str(out_emb), embeddings.astype(np.float16))
    out_lab = Path("/clade/ng/submission/ref_labels.json")
    with open(str(out_lab), "w") as f:
        json.dump(all_labels, f)
    print(f"Saved: {out_emb} ({out_emb.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved: {out_lab}")


if __name__ == "__main__":
    main()
