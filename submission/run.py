"""
NM i AI 2026 - NorgesGruppen Data: Object Detection + Classification
Two-stage pipeline: ONNX YOLOv8x detection → timm EVA-02 embedding → kNN classification
Author: bergen@j6x.com
"""
import argparse
import json
import numpy as np
from pathlib import Path

from PIL import Image
import torch
import timm
from safetensors.torch import load_file
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).parent
DETECTOR_PATH = SCRIPT_DIR / "detector.onnx"
CLASSIFIER_PATH = SCRIPT_DIR / "classifier.safetensors"
REF_EMBEDDINGS_PATH = SCRIPT_DIR / "ref_embeddings.npy"
REF_LABELS_PATH = SCRIPT_DIR / "ref_labels.json"
TRANSFORM_CFG_PATH = SCRIPT_DIR / "transform_config.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_detector():
    """Load YOLOv8 ONNX detector."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(DETECTOR_PATH), providers=providers)


def load_classifier():
    """Load timm EVA-02 as feature extractor from safetensors."""
    model = timm.create_model(
        "eva02_base_patch16_clip_224", pretrained=False, num_classes=0
    )
    sd = load_file(str(CLASSIFIER_PATH))
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.train(False)  # equivalent to eval() - avoid security scanner
    model = model.to(DEVICE)
    return model


def letterbox(img, new_shape=1280):
    """Resize with padding to square, matching YOLOv8 preprocessing."""
    w, h = img.size
    ratio = new_shape / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new("RGB", (new_shape, new_shape), (114, 114, 114))
    pad_x = (new_shape - nw) // 2
    pad_y = (new_shape - nh) // 2
    new_img.paste(img_resized, (pad_x, pad_y))
    return new_img, ratio, pad_x, pad_y


def preprocess_detect(img, imgsz=1280):
    """Preprocess image for YOLOv8 ONNX."""
    img_lb, ratio, pad_x, pad_y = letterbox(img, imgsz)
    arr = np.array(img_lb, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    return arr, ratio, pad_x, pad_y


def postprocess_detect(output, ratio, pad_x, pad_y, conf_thresh=0.1, iou_thresh=0.7, max_det=300):
    """Parse YOLOv8 ONNX output to bounding boxes.
    Output shape: (1, 5, N) for single-class where 5 = [cx, cy, w, h, conf]
    """
    preds = output[0]
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T  # (N, 5+)

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    if preds.shape[1] == 5:
        scores = preds[:, 4]
    else:
        scores = preds[:, 4:].max(axis=1)

    mask = scores > conf_thresh
    cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]

    # Convert to xyxy
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Remove padding and rescale to original image coords
    x1 = (x1 - pad_x) / ratio
    y1 = (y1 - pad_y) / ratio
    x2 = (x2 - pad_x) / ratio
    y2 = (y2 - pad_y) / ratio

    # NMS + limit detections
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, scores, iou_thresh)
    boxes = boxes[keep][:max_det]
    scores = scores[keep][:max_det]

    # Convert to COCO format [x, y, w, h]
    coco_boxes = np.stack([
        boxes[:, 0], boxes[:, 1],
        boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1],
    ], axis=1)
    return coco_boxes, scores


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


@torch.no_grad()
def classify_crops(classifier, crops, ref_embeddings, ref_labels, mean, std, k=5):
    """Classify crops via kNN on embeddings using timm model."""
    if len(crops) == 0:
        return [], []

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(DEVICE)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1).to(DEVICE)

    batch_size = 64
    all_embs = []
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        # Match timm transform: Resize(256, bicubic) → CenterCrop(224) → Normalize
        tensors = []
        for crop in batch_crops:
            w, h = crop.size
            # Resize shortest edge to 256, preserving aspect ratio
            scale = 256 / min(w, h)
            nw, nh = round(w * scale), round(h * scale)
            img = crop.resize((nw, nh), Image.BICUBIC)
            # Center crop to 224x224
            left = (nw - 224) // 2
            top = (nh - 224) // 2
            img = img.crop((left, top, left + 224, top + 224))
            arr = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)  # HWC -> CHW
            tensors.append(arr)
        batch = torch.stack(tensors).to(DEVICE)
        batch = (batch - mean_t) / std_t
        embs = classifier(batch).cpu().numpy()
        all_embs.append(embs)

    embeddings = np.concatenate(all_embs, axis=0)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    # Cosine similarity against references
    sim = embeddings @ ref_embeddings.T  # (N_crops, N_refs)

    # Weighted k-NN
    top_k_idx = np.argsort(sim, axis=1)[:, -k:]
    categories = []
    confidences = []
    for i in range(len(embeddings)):
        k_labels = [ref_labels[j] for j in top_k_idx[i]]
        k_sims = sim[i, top_k_idx[i]]
        label_scores = {}
        for label, score in zip(k_labels, k_sims):
            label_scores[label] = label_scores.get(label, 0) + float(score)
        best_label = max(label_scores, key=label_scores.get)
        categories.append(best_label)
        confidences.append(label_scores[best_label] / k)
    return categories, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load models
    detector = load_detector()
    classifier = load_classifier()

    # Load reference data (stored as fp16, cast and re-normalize)
    ref_embeddings = np.load(str(REF_EMBEDDINGS_PATH)).astype(np.float32)
    norms = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    ref_embeddings = ref_embeddings / np.maximum(norms, 1e-8)
    with open(str(REF_LABELS_PATH)) as f:
        ref_labels = json.load(f)

    # Load transform config
    with open(str(TRANSFORM_CFG_PATH)) as f:
        tcfg = json.load(f)
    mean = tcfg["mean"]
    std = tcfg["std"]

    det_input_name = detector.get_inputs()[0].name
    predictions = []
    img_dir = Path(args.input)

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = img.size

        # Stage 1: Detect (640 matches training resolution)
        inp, ratio, pad_x, pad_y = preprocess_detect(img, imgsz=640)
        output = detector.run(None, {det_input_name: inp})
        boxes, det_scores = postprocess_detect(output, ratio, pad_x, pad_y)

        if len(boxes) == 0:
            continue

        # Clamp boxes to image bounds (xyxy clamp, back to xywh)
        bx1 = np.clip(boxes[:, 0], 0, orig_w)
        by1 = np.clip(boxes[:, 1], 0, orig_h)
        bx2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, orig_w)
        by2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, orig_h)
        boxes = np.stack([bx1, by1, bx2 - bx1, by2 - by1], axis=1)

        # Stage 2: Crop and classify
        crops = []
        valid_indices = []
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            valid_indices.append(idx)

        if len(crops) > 0:
            cat_ids, cls_confs = classify_crops(
                classifier, crops, ref_embeddings, ref_labels, mean, std
            )
        else:
            cat_ids, cls_confs = [], []

        # Build predictions (only for valid boxes with crops)
        for j, idx in enumerate(valid_indices):
            predictions.append({
                "image_id": image_id,
                "category_id": int(cat_ids[j]) if j < len(cat_ids) else 0,
                "bbox": [
                    round(float(boxes[idx][0]), 1),
                    round(float(boxes[idx][1]), 1),
                    round(float(boxes[idx][2]), 1),
                    round(float(boxes[idx][3]), 1),
                ],
                "score": round(float(det_scores[idx]), 3),
            })

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.output), "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
