"""
Cache raw detector outputs for all images (pre-NMS).
Saves enough data to replay any NMS/threshold/M-cap sweep offline.

Usage:
  python cache_detections.py [--data /root/ng/data] [--detector /root/ng/detector.onnx]
"""
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import time


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


def preprocess(img, imgsz=1280):
    img_lb, ratio, pad_x, pad_y = letterbox(img, imgsz)
    arr = np.array(img_lb, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr, ratio, pad_x, pad_y


def decode_raw(output, ratio, pad_x, pad_y, conf_thresh=0.0001):
    """Decode YOLOv8 output to xyxy boxes + scores, very generous threshold."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/root/ng/data')
    parser.add_argument('--detector', default='/root/ng/detector.onnx')
    parser.add_argument('--cache_dir', default='/root/ng/cache')
    parser.add_argument('--conf_thresh', type=float, default=0.0001)
    parser.add_argument('--max_per_image', type=int, default=1500)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) / 'detections'
    cache_dir.mkdir(parents=True, exist_ok=True)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    detector = ort.InferenceSession(args.detector, providers=providers)
    input_name = detector.get_inputs()[0].name
    print(f"Detector loaded, providers: {detector.get_providers()}")

    img_dir = Path(args.data) / 'train' / 'images'
    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    t0 = time.time()
    for i, img_path in enumerate(images):
        image_id = int(img_path.stem.split('_')[-1])
        img = Image.open(str(img_path)).convert('RGB')
        orig_w, orig_h = img.size

        # Original
        inp, ratio, pad_x, pad_y = preprocess(img)
        output = detector.run(None, {input_name: inp})
        boxes_orig, scores_orig = decode_raw(output, ratio, pad_x, pad_y, args.conf_thresh)

        # Flip TTA
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        inp_f, ratio_f, pad_x_f, pad_y_f = preprocess(img_flip)
        output_f = detector.run(None, {input_name: inp_f})
        boxes_flip, scores_flip = decode_raw(output_f, ratio_f, pad_x_f, pad_y_f, args.conf_thresh)

        # Mirror flip boxes back
        if len(boxes_flip) > 0:
            boxes_flip_m = boxes_flip.copy()
            boxes_flip_m[:, 0] = orig_w - boxes_flip[:, 2]
            boxes_flip_m[:, 2] = orig_w - boxes_flip[:, 0]
        else:
            boxes_flip_m = boxes_flip

        # Concatenate TTA results (pre-NMS)
        parts_b = [b for b in [boxes_orig, boxes_flip_m] if len(b) > 0]
        parts_s = [s for s in [scores_orig, scores_flip] if len(s) > 0]
        if parts_b:
            all_boxes = np.concatenate(parts_b, axis=0)
            all_scores = np.concatenate(parts_s, axis=0)
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_scores = np.zeros(0, dtype=np.float32)

        # Clamp to image bounds
        if len(all_boxes) > 0:
            all_boxes[:, 0] = np.clip(all_boxes[:, 0], 0, orig_w)
            all_boxes[:, 1] = np.clip(all_boxes[:, 1], 0, orig_h)
            all_boxes[:, 2] = np.clip(all_boxes[:, 2], 0, orig_w)
            all_boxes[:, 3] = np.clip(all_boxes[:, 3], 0, orig_h)

        # Sort by score, keep top N
        if len(all_scores) > args.max_per_image:
            top_idx = np.argsort(all_scores)[::-1][:args.max_per_image]
            all_boxes = all_boxes[top_idx]
            all_scores = all_scores[top_idx]

        # Convert xyxy to xywh
        if len(all_boxes) > 0:
            boxes_xywh = np.stack([
                all_boxes[:, 0], all_boxes[:, 1],
                all_boxes[:, 2] - all_boxes[:, 0],
                all_boxes[:, 3] - all_boxes[:, 1]
            ], axis=1)
            # Filter zero-area
            valid = (boxes_xywh[:, 2] > 0) & (boxes_xywh[:, 3] > 0)
            all_boxes = all_boxes[valid]
            boxes_xywh = boxes_xywh[valid]
            all_scores = all_scores[valid]
        else:
            boxes_xywh = np.zeros((0, 4), dtype=np.float32)

        np.savez_compressed(
            str(cache_dir / f'{image_id}.npz'),
            boxes_xyxy=all_boxes.astype(np.float32),
            boxes_xywh=boxes_xywh.astype(np.float32),
            det_scores=all_scores.astype(np.float32),
            orig_w=np.int32(orig_w),
            orig_h=np.int32(orig_h),
            image_id=np.int32(image_id),
        )

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(images) - i - 1) / rate
            print(f"  [{i+1}/{len(images)}] img {image_id}: {len(all_scores)} boxes | "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    print(f"\nDone. Cached {len(images)} images in {time.time()-t0:.1f}s → {cache_dir}")


if __name__ == '__main__':
    main()
