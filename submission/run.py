"""
NM i AI 2026 - NorgesGruppen Data: Object Detection + Classification
Two-stage pipeline: ONNX YOLOv8x 1280px detection → timm EVA-02 softmax classifier
v5.2: Top-K + 2-view classification TTA (orig + scale224 logit averaging)
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EVA-02 CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
NUM_CLASSES = 356


def load_detector():
    """Load YOLOv8 ONNX detector."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(DETECTOR_PATH), providers=providers)


def load_classifier():
    """Load fine-tuned EVA-02 with 356-class classification head."""
    model = timm.create_model(
        "eva02_base_patch16_clip_224", pretrained=False, num_classes=NUM_CLASSES
    )
    sd = load_file(str(CLASSIFIER_PATH))
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.train(False)
    model = model.half().to(DEVICE)
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


def postprocess_detect_raw(output, ratio, pad_x, pad_y, conf_thresh=0.001):
    """Parse YOLOv8 ONNX output to raw xyxy boxes (no NMS)."""
    preds = output[0]
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    if preds.shape[1] == 5:
        scores = preds[:, 4]
    else:
        scores = preds[:, 4:].max(axis=1)

    mask = scores > conf_thresh
    cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]

    x1 = (cx - w / 2 - pad_x) / ratio
    y1 = (cy - h / 2 - pad_y) / ratio
    x2 = (cx + w / 2 - pad_x) / ratio
    y2 = (cy + h / 2 - pad_y) / ratio

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, scores


def detect_multiscale(detector, det_input_name, img):
    """Detection TTA: full image + horizontal flip, merge with NMS."""
    all_boxes = []
    all_scores = []
    orig_w = img.size[0]

    # Original
    inp, ratio, pad_x, pad_y = preprocess_detect(img, imgsz=1280)
    output = detector.run(None, {det_input_name: inp})
    boxes, scores = postprocess_detect_raw(output, ratio, pad_x, pad_y)
    all_boxes.append(boxes)
    all_scores.append(scores)

    # Horizontal flip
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    inp_f, ratio_f, pad_x_f, pad_y_f = preprocess_detect(img_flip, imgsz=1280)
    output_f = detector.run(None, {det_input_name: inp_f})
    boxes_f, scores_f = postprocess_detect_raw(output_f, ratio_f, pad_x_f, pad_y_f)
    if len(boxes_f) > 0:
        boxes_f_m = boxes_f.copy()
        boxes_f_m[:, 0] = orig_w - boxes_f[:, 2]
        boxes_f_m[:, 2] = orig_w - boxes_f[:, 0]
        all_boxes.append(boxes_f_m)
        all_scores.append(scores_f)

    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.zeros(0)

    all_boxes = np.concatenate([b for b in all_boxes if len(b) > 0], axis=0)
    all_scores = np.concatenate([s for s in all_scores if len(s) > 0], axis=0)

    # Global NMS
    keep = nms(all_boxes, all_scores, 0.5)
    boxes = all_boxes[keep][:500]
    scores = all_scores[keep][:500]

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


def _prepare_crop(crop, resize_short_edge=256):
    """Resize(short_edge, bicubic) → CenterCrop(224) → float tensor."""
    w, h = crop.size
    scale = resize_short_edge / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = crop.resize((nw, nh), Image.BICUBIC)
    left = (nw - 224) // 2
    top = (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    return torch.from_numpy(
        np.array(img, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)  # HWC -> CHW


TOP_K = 15           # Submit top-K category predictions per detection
SCORE_DECAY = 0.7    # Score decay factor per rank
TEMPERATURE = 0.9    # Softmax temperature (OOF-tuned)
ALPHA = 0.4          # Classification weight in additive fusion


TTA_VIEWS = [256, 224]  # resize_short_edge values for classification TTA


@torch.no_grad()
def classify_crops(classifier, crops):
    """Classify crops with 2-view TTA (orig 256 + scale224), logit averaging.
    v5.2: OOF-validated +0.001 blend on top of v5.1's top-K."""
    if len(crops) == 0:
        return [], []

    mean_t = torch.tensor(CLIP_MEAN, dtype=torch.float16).view(1, 3, 1, 1).to(DEVICE)
    std_t = torch.tensor(CLIP_STD, dtype=torch.float16).view(1, 3, 1, 1).to(DEVICE)

    batch_size = 64
    # Accumulate logits across TTA views
    all_logits = None

    for resize_edge in TTA_VIEWS:
        view_logits = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            tensors = [_prepare_crop(c, resize_short_edge=resize_edge) for c in batch_crops]
            batch = torch.stack(tensors).half().to(DEVICE)
            batch = (batch - mean_t) / std_t
            features = classifier.forward_features(batch)
            logits = classifier.forward_head(features)
            view_logits.append(logits.float())

        view_all = torch.cat(view_logits, dim=0)  # [N, NUM_CLASSES]
        if all_logits is None:
            all_logits = view_all
        else:
            all_logits = torch.maximum(all_logits, view_all)

    # Max-logit aggregation, then temperature-scaled softmax
    probs = torch.softmax(all_logits / TEMPERATURE, dim=1)
    topk_confs, topk_cats = probs.topk(TOP_K, dim=1)

    return topk_cats.cpu().tolist(), topk_confs.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load models
    detector = load_detector()
    classifier = load_classifier()

    det_input_name = detector.get_inputs()[0].name
    predictions = []
    img_dir = Path(args.input)

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = img.size

        # Stage 1: Detection TTA (full + flip)
        boxes, det_scores = detect_multiscale(detector, det_input_name, img)

        if len(boxes) == 0:
            continue

        # Clamp boxes to image bounds (xyxy clamp, back to xywh)
        bx1 = np.clip(boxes[:, 0], 0, orig_w)
        by1 = np.clip(boxes[:, 1], 0, orig_h)
        bx2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, orig_w)
        by2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, orig_h)
        boxes = np.stack([bx1, by1, bx2 - bx1, by2 - by1], axis=1)

        # Cap: only classify top-M detections per image by det score
        MAX_CLASSIFY = 300
        if len(boxes) > MAX_CLASSIFY:
            top_idx = np.argsort(det_scores)[::-1][:MAX_CLASSIFY]
            boxes = boxes[top_idx]
            det_scores = det_scores[top_idx]

        # Stage 2: Crop and classify (pure softmax, no kNN)
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
            cat_ids, cls_confs = classify_crops(classifier, crops)
        else:
            cat_ids, cls_confs = [], []

        # Score: top-K category predictions with decayed scores
        # v5.1: OOF-validated +0.012 improvement over single-category predictions
        img_preds = []
        for j, idx in enumerate(valid_indices):
            det_s = float(det_scores[idx])
            bbox = [
                round(float(boxes[idx][0]), 1),
                round(float(boxes[idx][1]), 1),
                round(float(boxes[idx][2]), 1),
                round(float(boxes[idx][3]), 1),
            ]
            if j < len(cat_ids):
                topk_c = cat_ids[j]    # list of K category ids
                topk_s = cls_confs[j]  # list of K confidences
                for rank in range(len(topk_c)):
                    cls_s = float(topk_s[rank])
                    score = (1 - ALPHA) * det_s + ALPHA * cls_s
                    if rank > 0:
                        score *= SCORE_DECAY ** rank
                    if score < 0.01:
                        break
                    img_preds.append({
                        "image_id": image_id,
                        "category_id": int(topk_c[rank]),
                        "bbox": bbox,
                        "score": round(score, 4),
                    })
            else:
                img_preds.append({
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "score": round(det_s, 4),
                })

        # Keep top predictions per image (safe output size, COCO maxDets=100)
        img_preds.sort(key=lambda p: p["score"], reverse=True)
        predictions.extend(img_preds[:500])

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.output), "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
