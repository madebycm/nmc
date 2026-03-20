"""
Modified run.py that saves raw det_score and cls_score for offline fusion sweep.
Also tests: no-flip classifier TTA, crop padding, score fusion variants.
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

SCRIPT_DIR = Path(__file__).parent / "submission"
DETECTOR_PATH = SCRIPT_DIR / "detector.onnx"
CLASSIFIER_PATH = SCRIPT_DIR / "classifier.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
NUM_CLASSES = 356


def load_detector():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(DETECTOR_PATH), providers=providers)


def load_classifier():
    model = timm.create_model("eva02_base_patch16_clip_224", pretrained=False, num_classes=NUM_CLASSES)
    sd = load_file(str(CLASSIFIER_PATH))
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.train(False)
    model = model.to(DEVICE)
    return model


def letterbox(img, new_shape=1280):
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
    img_lb, ratio, pad_x, pad_y = letterbox(img, imgsz)
    arr = np.array(img_lb, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr, ratio, pad_x, pad_y


def postprocess_detect_raw(output, ratio, pad_x, pad_y, conf_thresh=0.05):
    preds = output[0]
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    scores = preds[:, 4] if preds.shape[1] == 5 else preds[:, 4:].max(axis=1)
    mask = scores > conf_thresh
    cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]
    x1 = (cx - w / 2 - pad_x) / ratio
    y1 = (cy - h / 2 - pad_y) / ratio
    x2 = (cx + w / 2 - pad_x) / ratio
    y2 = (cy + h / 2 - pad_y) / ratio
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, scores


def nms(boxes, scores, iou_threshold):
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


def detect_with_flip(detector, det_input_name, img):
    all_boxes, all_scores = [], []
    orig_w = img.size[0]

    inp, ratio, pad_x, pad_y = preprocess_detect(img, imgsz=1280)
    output = detector.run(None, {det_input_name: inp})
    boxes, scores = postprocess_detect_raw(output, ratio, pad_x, pad_y)
    all_boxes.append(boxes)
    all_scores.append(scores)

    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    inp_f, ratio_f, pad_x_f, pad_y_f = preprocess_detect(img_flip, imgsz=1280)
    output_f = detector.run(None, {det_input_name: inp_f})
    boxes_f, scores_f = postprocess_detect_raw(output_f, ratio_f, pad_x_f, pad_y_f)
    if len(boxes_f) > 0:
        bm = boxes_f.copy()
        bm[:, 0] = orig_w - boxes_f[:, 2]
        bm[:, 2] = orig_w - boxes_f[:, 0]
        all_boxes.append(bm)
        all_scores.append(scores_f)

    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.zeros(0)

    all_boxes = np.concatenate([b for b in all_boxes if len(b) > 0])
    all_scores = np.concatenate([s for s in all_scores if len(s) > 0])
    keep = nms(all_boxes, all_scores, 0.7)
    boxes = all_boxes[keep][:500]
    scores = all_scores[keep][:500]
    coco_boxes = np.stack([
        boxes[:, 0], boxes[:, 1],
        boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1],
    ], axis=1)
    return coco_boxes, scores


def _prepare_crop(crop):
    w, h = crop.size
    scale = 256 / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = crop.resize((nw, nh), Image.BICUBIC)
    left = (nw - 224) // 2
    top = (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def _prepare_crop_padded(crop):
    """Square-pad then resize to 224 — preserves full box content."""
    w, h = crop.size
    s = max(w, h)
    padded = Image.new("RGB", (s, s), (114, 114, 114))
    padded.paste(crop, ((s - w) // 2, (s - h) // 2))
    padded = padded.resize((224, 224), Image.BICUBIC)
    return torch.from_numpy(np.array(padded, dtype=np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def classify_crops(classifier, crops, use_flip=True, use_padded=False):
    if len(crops) == 0:
        return [], []
    prep = _prepare_crop_padded if use_padded else _prepare_crop
    mean_t = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(DEVICE)
    std_t = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1).to(DEVICE)
    batch_size = 64
    all_cats, all_confs = [], []
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        tensors = [prep(c) for c in batch_crops]
        batch = torch.stack(tensors).to(DEVICE)
        batch = (batch - mean_t) / std_t
        logits = classifier(batch)
        if use_flip:
            logits_flip = classifier(batch.flip(-1))
            logits = (logits + logits_flip) / 2
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_cats.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())
    return all_cats, all_confs


def run_pipeline(detector, classifier, img_dir, use_flip_cls=True, use_padded=False):
    det_input_name = detector.get_inputs()[0].name
    results = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = img.size

        boxes, det_scores = detect_with_flip(detector, det_input_name, img)
        if len(boxes) == 0:
            continue

        bx1 = np.clip(boxes[:, 0], 0, orig_w)
        by1 = np.clip(boxes[:, 1], 0, orig_h)
        bx2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, orig_w)
        by2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, orig_h)
        boxes = np.stack([bx1, by1, bx2 - bx1, by2 - by1], axis=1)

        crops, valid_indices = [], []
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            valid_indices.append(idx)

        if len(crops) > 0:
            cat_ids, cls_confs = classify_crops(classifier, crops, use_flip=use_flip_cls, use_padded=use_padded)
        else:
            cat_ids, cls_confs = [], []

        for j, idx in enumerate(valid_indices):
            results.append({
                "image_id": image_id,
                "category_id": int(cat_ids[j]) if j < len(cat_ids) else 0,
                "bbox": [round(float(boxes[idx][k]), 1) for k in range(4)],
                "det_score": round(float(det_scores[idx]), 4),
                "cls_score": round(float(cls_confs[j]) if j < len(cls_confs) else 0.0, 4),
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-flip-cls", action="store_true")
    parser.add_argument("--padded-crops", action="store_true")
    args = parser.parse_args()

    detector = load_detector()
    classifier = load_classifier()
    img_dir = Path(args.input)

    results = run_pipeline(
        detector, classifier, img_dir,
        use_flip_cls=not args.no_flip_cls,
        use_padded=args.padded_crops,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.output), "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
