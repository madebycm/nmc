"""
Overnight evaluation lab: classify cached detections, ceiling diagnostics, parametric sweep.

Usage:
  python overnight_eval.py --weights /root/ng/classifier.safetensors --tag alldata --fold 1
  python overnight_eval.py --weights /root/ng/output_ceonly/classifier_arcface_best.safetensors --tag ceonly_f1 --fold 1
  python overnight_eval.py --skip_classify --tag ceonly_f1 --fold 1   # reuse cached classifications
"""
import argparse
import copy
import io
import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import timm
from safetensors.torch import load_file
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NUM_CLASSES = 356
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ─── Fold split (must match train_arcface.py exactly) ───────────────────────

def get_fold_split(ann_path, fold=1, n_folds=5, seed=42):
    with open(ann_path) as f:
        coco = json.load(f)
    image_ids = sorted(set(img['id'] for img in coco['images']))
    rng = np.random.RandomState(seed)
    rng.shuffle(image_ids)
    fold_size = len(image_ids) // n_folds
    val_start = (fold - 1) * fold_size
    val_end = val_start + fold_size if fold < n_folds else len(image_ids)
    val_ids = image_ids[val_start:val_end]
    train_ids = [i for i in image_ids if i not in set(val_ids)]
    return train_ids, val_ids


# ─── NMS ────────────────────────────────────────────────────────────────────

def hard_nms(boxes_xyxy, scores, iou_thresh):
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
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
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


def soft_nms(boxes_xyxy, scores, sigma=0.5, score_thresh=0.001):
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    N = len(boxes_xyxy)
    boxes = boxes_xyxy.copy()
    sc = scores.copy()
    indices = np.arange(N)
    for i in range(N):
        max_pos = i + np.argmax(sc[i:])
        boxes[[i, max_pos]] = boxes[[max_pos, i]]
        sc[[i, max_pos]] = sc[[max_pos, i]]
        indices[[i, max_pos]] = indices[[max_pos, i]]
        if i + 1 >= N:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        areas = (boxes[i+1:, 2] - boxes[i+1:, 0]) * (boxes[i+1:, 3] - boxes[i+1:, 1])
        iou = inter / (area_i + areas - inter + 1e-8)
        sc[i+1:] *= np.exp(-(iou ** 2) / sigma)
    keep = sc > score_thresh
    return indices[keep], sc[keep]


# ─── IoU ────────────────────────────────────────────────────────────────────

def compute_iou_matrix(pred_xywh, gt_xywh):
    px1, py1 = pred_xywh[:, 0], pred_xywh[:, 1]
    px2, py2 = px1 + pred_xywh[:, 2], py1 + pred_xywh[:, 3]
    gx1, gy1 = gt_xywh[:, 0], gt_xywh[:, 1]
    gx2, gy2 = gx1 + gt_xywh[:, 2], gy1 + gt_xywh[:, 3]
    ix1 = np.maximum(px1[:, None], gx1[None, :])
    iy1 = np.maximum(py1[:, None], gy1[None, :])
    ix2 = np.minimum(px2[:, None], gx2[None, :])
    iy2 = np.minimum(py2[:, None], gy2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    return inter / (pred_area[:, None] + gt_area[None, :] - inter + 1e-8)


# ─── Classifier ─────────────────────────────────────────────────────────────

def prepare_crop(crop):
    w, h = crop.size
    scale = 256 / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = crop.resize((nw, nh), Image.BICUBIC)
    left = (nw - 224) // 2
    top = (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_classifier(weights_path, device):
    model = timm.create_model('eva02_base_patch16_clip_224', pretrained=False, num_classes=NUM_CLASSES)
    sd = load_file(str(weights_path))
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.train(False)
    return model.half().to(device)


@torch.no_grad()
def classify_cached_boxes(model, img_dir, id_to_file, det_cache_dir, image_ids, device, batch_size=64):
    mean_t = torch.tensor(CLIP_MEAN, dtype=torch.float16).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(CLIP_STD, dtype=torch.float16).view(1, 3, 1, 1).to(device)
    results = {}
    total = 0
    t0 = time.time()
    for idx, image_id in enumerate(sorted(image_ids)):
        det_path = det_cache_dir / f'{image_id}.npz'
        if not det_path.exists():
            continue
        det = np.load(str(det_path))
        boxes_xyxy = det['boxes_xyxy']
        if len(boxes_xyxy) == 0:
            results[image_id] = {'logits': np.zeros((0, NUM_CLASSES), dtype=np.float16),
                                  'features': np.zeros((0, 768), dtype=np.float16),
                                  'valid_idx': np.array([], dtype=np.int32)}
            continue
        img = Image.open(str(img_dir / id_to_file[image_id])).convert('RGB')
        crops, valid_idx = [], []
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(prepare_crop(img.crop((x1, y1, x2, y2))))
            valid_idx.append(i)
        if not crops:
            results[image_id] = {'logits': np.zeros((0, NUM_CLASSES), dtype=np.float16),
                                  'features': np.zeros((0, 768), dtype=np.float16),
                                  'valid_idx': np.array([], dtype=np.int32)}
            continue
        all_logits, all_features = [], []
        for bi in range(0, len(crops), batch_size):
            batch = torch.stack(crops[bi:bi+batch_size]).half().to(device)
            batch = (batch - mean_t) / std_t
            features = model.forward_features(batch)
            logits = model.forward_head(features)
            cls_feat = features[:, 0].float()
            cls_feat = cls_feat / (cls_feat.norm(dim=1, keepdim=True) + 1e-8)
            all_logits.append(logits.cpu().half().numpy())
            all_features.append(cls_feat.cpu().half().numpy())
        results[image_id] = {'logits': np.concatenate(all_logits, 0),
                              'features': np.concatenate(all_features, 0),
                              'valid_idx': np.array(valid_idx, dtype=np.int32)}
        total += len(valid_idx)
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(image_ids)}] {total} boxes, {time.time()-t0:.0f}s")
    print(f"  Total: {total} boxes in {time.time()-t0:.1f}s")
    return results


@torch.no_grad()
def build_ref_embeddings(model, ann_path, img_dir, train_ids, data_dir, device, batch_size=64):
    with open(ann_path) as f:
        coco = json.load(f)
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    train_set = set(train_ids)
    mean_t = torch.tensor(CLIP_MEAN, dtype=torch.float16).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(CLIP_STD, dtype=torch.float16).view(1, 3, 1, 1).to(device)
    tensors, labels = [], []
    # Group annotations by image to avoid re-opening the same file
    anns_by_img = defaultdict(list)
    for ann in coco['annotations']:
        if ann['image_id'] in train_set:
            anns_by_img[ann['image_id']].append(ann)
    for img_id in sorted(anns_by_img.keys()):
        img = Image.open(str(img_dir / id_to_file[img_id])).convert('RGB')
        for ann in anns_by_img[img_id]:
            bbox = ann['bbox']
            if bbox[2] < 2 or bbox[3] < 2:
                continue
            x, y, w, h = bbox
            tensors.append(prepare_crop(img.crop((int(x), int(y), int(x+w), int(y+h)))))
            labels.append(ann['category_id'])
    print(f"  Crop embeddings: {len(tensors)} from {len(train_set)} train images")
    # Product images (always included)
    ean_path = Path(data_dir) / 'ean_to_catid.json'
    if ean_path.exists():
        with open(ean_path) as f:
            ean_map = json.load(f)
        for ean_dir in sorted(Path(data_dir).iterdir()):
            if not ean_dir.is_dir():
                continue
            cat_id = ean_map.get(ean_dir.name)
            if cat_id is None:
                continue
            for p in ean_dir.iterdir():
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    tensors.append(prepare_crop(Image.open(str(p)).convert('RGB')))
                    labels.append(cat_id)
        print(f"  With products: {len(tensors)} total")
    all_emb = []
    for bi in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[bi:bi+batch_size]).half().to(device)
        batch = (batch - mean_t) / std_t
        features = model.forward_features(batch)
        cls_feat = features[:, 0].float()
        cls_feat = cls_feat / (cls_feat.norm(dim=1, keepdim=True) + 1e-8)
        all_emb.append(cls_feat.cpu().numpy())
    ref_emb = np.concatenate(all_emb, 0).astype(np.float32)
    ref_labels = np.array(labels, dtype=np.int64)
    ref_counts = np.array([max((ref_labels == c).sum(), 1) for c in range(NUM_CLASSES)], dtype=np.float32)
    print(f"  Ref bank: {ref_emb.shape[0]} emb, {len(set(labels))} classes")
    return ref_emb, ref_labels, ref_counts


# ─── Evaluation ─────────────────────────────────────────────────────────────

def proxy_eval(preds_by_image, gt_by_image, val_ids):
    total_gt, total_det, total_correct = 0, 0, 0
    for img_id in val_ids:
        gt = gt_by_image.get(img_id, [])
        preds = sorted(preds_by_image.get(img_id, []), key=lambda x: x['score'], reverse=True)[:100]
        total_gt += len(gt)
        if not preds or not gt:
            continue
        gt_boxes = np.array([g['bbox'] for g in gt])
        gt_cats = [g['category_id'] for g in gt]
        pred_boxes = np.array([p['bbox'] for p in preds])
        iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
        matched = set()
        for pi in range(len(preds)):
            best_iou, best_gi = 0, -1
            for gi in range(len(gt)):
                if gi not in matched and iou_mat[pi, gi] > best_iou:
                    best_iou = iou_mat[pi, gi]
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched.add(best_gi)
                total_det += 1
                if preds[pi]['category_id'] == gt_cats[best_gi]:
                    total_correct += 1
    det_r = total_det / max(total_gt, 1)
    cls_a = total_correct / max(total_det, 1)
    return 0.7 * det_r + 0.3 * cls_a, det_r, cls_a


def full_eval(predictions, gt_coco_data, val_ids):
    if not predictions:
        return 0.0, 0.0, 0.0
    val_set = set(val_ids)

    def _eval(gt_data, preds):
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            coco_gt = COCO()
            coco_gt.dataset = gt_data
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(preds)
            ev = COCOeval(coco_gt, coco_dt, 'bbox')
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            return ev.stats[1]
        except Exception as e:
            sys.stdout = old
            print(f"  eval error: {e}")
            return 0.0
        finally:
            sys.stdout = old

    # Det eval (single-class)
    gt_det = copy.deepcopy(gt_coco_data)
    gt_det['annotations'] = [a for a in gt_det['annotations'] if a['image_id'] in val_set]
    for a in gt_det['annotations']:
        a['category_id'] = 1
    gt_det['categories'] = [{'id': 1, 'name': 'object'}]
    gt_det['images'] = [img for img in gt_det['images'] if img['id'] in val_set]
    det_preds = [{'image_id': p['image_id'], 'category_id': 1,
                  'bbox': p['bbox'], 'score': p['score']} for p in predictions]
    det_map50 = _eval(gt_det, det_preds)

    # Cls eval (multi-class)
    gt_cls = copy.deepcopy(gt_coco_data)
    gt_cls['annotations'] = [a for a in gt_cls['annotations'] if a['image_id'] in val_set]
    gt_cls['images'] = [img for img in gt_cls['images'] if img['id'] in val_set]
    cls_map50 = _eval(gt_cls, predictions)

    return det_map50, cls_map50, 0.7 * det_map50 + 0.3 * cls_map50


# ─── Ceilings ───────────────────────────────────────────────────────────────

def run_ceilings(det_data, cls_data, gt_by_image, val_ids, gt_coco_data):
    print("\n" + "=" * 60)
    print("CEILING DIAGNOSTICS")
    print("=" * 60)

    # A. Detector recall
    print("\n--- Detector Recall (IoU > 0.5) ---")
    for topk in [50, 100, 200, 300, 500, 1000]:
        total_gt, total_matched = 0, 0
        for img_id in val_ids:
            gt = gt_by_image.get(img_id, [])
            total_gt += len(gt)
            if not gt or img_id not in det_data:
                continue
            det = det_data[img_id]
            scores, boxes = det['det_scores'], det['boxes_xywh']
            if len(scores) > topk:
                idx = np.argsort(scores)[::-1][:topk]
                boxes = boxes[idx]
            if len(boxes) == 0:
                continue
            gt_boxes = np.array([g['bbox'] for g in gt])
            iou_mat = compute_iou_matrix(boxes, gt_boxes)
            matched = set()
            for pi in np.argsort(-iou_mat.max(axis=1)):
                for gi in np.argsort(-iou_mat[pi]):
                    if gi not in matched and iou_mat[pi, gi] >= 0.5:
                        matched.add(gi)
                        break
            total_matched += len(matched)
        print(f"  Top-{topk:4d}: recall={total_matched/max(total_gt,1):.4f} ({total_matched}/{total_gt})")

    # B. Oracle classification
    print("\n--- Oracle Classification (GT labels, NMS=0.5, conf=0.001, M=300) ---")
    oracle_preds = []
    for img_id in val_ids:
        gt = gt_by_image.get(img_id, [])
        if not gt or img_id not in det_data:
            continue
        det = det_data[img_id]
        scores, boxes_xyxy, boxes_xywh = det['det_scores'], det['boxes_xyxy'], det['boxes_xywh']
        mask = scores > 0.001
        if mask.sum() == 0:
            continue
        keep = hard_nms(boxes_xyxy[mask], scores[mask], 0.5)
        if len(keep) == 0:
            continue
        sel = np.where(mask)[0][keep][:300]
        gt_boxes = np.array([g['bbox'] for g in gt])
        gt_cats = [g['category_id'] for g in gt]
        xywh_sel = boxes_xywh[sel]
        iou_mat = compute_iou_matrix(xywh_sel, gt_boxes)
        matched_gt = set()
        for pi in range(len(sel)):
            best_iou, best_gi = 0, -1
            for gi in range(len(gt)):
                if gi not in matched_gt and iou_mat[pi, gi] > best_iou:
                    best_iou = iou_mat[pi, gi]
                    best_gi = gi
            cat = gt_cats[best_gi] if best_iou >= 0.5 and best_gi >= 0 else 0
            if best_iou >= 0.5 and best_gi >= 0:
                matched_gt.add(best_gi)
            oracle_preds.append({
                'image_id': img_id, 'category_id': cat,
                'bbox': [round(float(x), 1) for x in xywh_sel[pi]],
                'score': round(float(scores[sel[pi]]), 4)})
    det_m, cls_m, blend = full_eval(oracle_preds, gt_coco_data, val_ids)
    print(f"  Oracle: det={det_m:.4f} cls={cls_m:.4f} blend={blend:.4f}")
    return det_m, cls_m, blend


# ─── Prediction generation ──────────────────────────────────────────────────

def _apply_nms_and_cap(det, conf_thresh, nms_iou, nms_type, M):
    """Apply conf filter, NMS, M cap. Returns (orig_indices, det_scores, boxes_xywh)."""
    scores = det['det_scores']
    mask = scores > conf_thresh
    if mask.sum() == 0:
        return None
    b_xyxy = det['boxes_xyxy'][mask]
    sc = scores[mask]
    if nms_type == 'hard':
        keep = hard_nms(b_xyxy, sc, nms_iou)
        if len(keep) == 0:
            return None
        orig_idx = np.where(mask)[0][keep]
        sc = sc[keep]
    else:
        sigma = float(nms_type.split('_')[1])
        keep_idx, new_sc = soft_nms(b_xyxy, sc, sigma=sigma)
        if len(keep_idx) == 0:
            return None
        orig_idx = np.where(mask)[0][keep_idx]
        sc = new_sc
    if len(sc) > M:
        top = np.argsort(sc)[::-1][:M]
        orig_idx = orig_idx[top]
        sc = sc[top]
    return orig_idx, sc, det['boxes_xywh'][orig_idx]


def _get_cls_pred(logits_row, temp):
    log = logits_row.astype(np.float32)
    if temp != 1.0:
        log = log / temp
    log -= log.max()
    probs = np.exp(log)
    probs /= probs.sum()
    cat = int(np.argmax(probs))
    return cat, float(probs[cat]), probs


def generate_predictions(det_data, cls_data, val_ids, config):
    score_mode = config.get('score_mode', 'mul')
    alpha = config['alpha']
    temp = config['temp']
    preds, preds_by_img = [], defaultdict(list)
    for img_id in val_ids:
        if img_id not in det_data or img_id not in cls_data:
            continue
        result = _apply_nms_and_cap(det_data[img_id], config['conf_thresh'],
                                     config['nms_iou'], config['nms_type'], config['M'])
        if result is None:
            continue
        orig_idx, det_sc, boxes_xywh = result
        cls = cls_data[img_id]
        valid_map = {int(v): ci for ci, v in enumerate(cls['valid_idx'])}
        for j, oidx in enumerate(orig_idx):
            ci = valid_map.get(int(oidx))
            if ci is None:
                continue
            cat, conf, _ = _get_cls_pred(cls['logits'][ci], temp)
            ds = float(det_sc[j])
            if score_mode == 'det' or alpha == 0:
                score = ds
            elif score_mode == 'mul':
                score = ds * (conf ** alpha)
            elif score_mode == 'add':
                score = (1 - alpha) * ds + alpha * conf
            else:
                score = ds
            p = {'image_id': img_id, 'category_id': cat,
                 'bbox': [round(float(x), 1) for x in boxes_xywh[j]],
                 'score': round(float(score), 6)}
            preds.append(p)
            preds_by_img[img_id].append(p)
    return preds, preds_by_img


def generate_predictions_knn(det_data, cls_data, val_ids, config, ref_emb, ref_labels, ref_counts):
    alpha = config['alpha']
    temp = config['temp']
    rt = config['route_thresh']
    ka = config['knn_alpha']
    kt = config['knn_temp']
    rp = config.get('route_penalty', 1.0)
    preds, preds_by_img = [], defaultdict(list)
    for img_id in val_ids:
        if img_id not in det_data or img_id not in cls_data:
            continue
        result = _apply_nms_and_cap(det_data[img_id], config['conf_thresh'],
                                     config['nms_iou'], config['nms_type'], config['M'])
        if result is None:
            continue
        orig_idx, det_sc, boxes_xywh = result
        cls = cls_data[img_id]
        valid_map = {int(v): ci for ci, v in enumerate(cls['valid_idx'])}
        for j, oidx in enumerate(orig_idx):
            ci = valid_map.get(int(oidx))
            if ci is None:
                continue
            cat, soft_conf, probs = _get_cls_pred(cls['logits'][ci], temp)
            penalty = 1.0
            if soft_conf < rt:
                feat = cls['features'][ci].astype(np.float32)
                feat /= (np.linalg.norm(feat) + 1e-8)
                sims = feat @ ref_emb.T
                class_sum = np.zeros(NUM_CLASSES, dtype=np.float32)
                np.add.at(class_sum, ref_labels, sims)
                class_mean = class_sum / ref_counts
                kl = class_mean / kt
                kl -= kl.max()
                knn_probs = np.exp(kl)
                knn_probs /= knn_probs.sum()
                blended = ka * probs + (1 - ka) * knn_probs
                cat = int(np.argmax(blended))
                soft_conf = float(blended[cat])
                penalty = rp
            ds = float(det_sc[j])
            if alpha > 0:
                score = ds * (soft_conf ** alpha) * penalty
            else:
                score = ds * penalty
            p = {'image_id': img_id, 'category_id': cat,
                 'bbox': [round(float(x), 1) for x in boxes_xywh[j]],
                 'score': round(float(score), 6)}
            preds.append(p)
            preds_by_img[img_id].append(p)
    return preds, preds_by_img


# ─── Sweep ──────────────────────────────────────────────────────────────────

def sweep_stage_a(det_data, cls_data, gt_by_image, gt_coco_data, val_ids, output_path):
    print("\n" + "=" * 60)
    print("STAGE A SWEEP: No kNN")
    print("=" * 60)
    confs = [0.001, 0.003, 0.01, 0.03]
    nms_ious = [0.45, 0.50, 0.55, 0.60, 0.65]
    nms_types = ['hard', 'soft_0.5']
    Ms = [100, 200, 300, 500]
    alphas = [0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
    temps = [1.0, 1.5, 2.0]
    results = []
    n = sum(len(confs)*len(nms_ious)*len(nms_types)*len(Ms)*(1 if a == 0 else len(temps)) for a in alphas)
    print(f"  {n} configs")
    t0 = time.time()
    done = 0
    for conf in confs:
        for nms_iou in nms_ious:
            for nms_type in nms_types:
                for M in Ms:
                    for alpha in alphas:
                        ts = [1.0] if alpha == 0 else temps
                        for temp in ts:
                            cfg = {'conf_thresh': conf, 'nms_iou': nms_iou, 'nms_type': nms_type,
                                   'M': M, 'alpha': alpha, 'temp': temp, 'score_mode': 'mul'}
                            _, pbi = generate_predictions(det_data, cls_data, val_ids, cfg)
                            px, dr, ca = proxy_eval(pbi, gt_by_image, val_ids)
                            results.append({**cfg, 'proxy': px, 'det_recall': dr, 'cls_acc': ca})
                            done += 1
                            if done % 200 == 0:
                                best = max(r['proxy'] for r in results)
                                print(f"  [{done}/{n}] {time.time()-t0:.0f}s | best proxy: {best:.4f}")
    results.sort(key=lambda x: x['proxy'], reverse=True)
    print(f"\n  Proxy done in {time.time()-t0:.1f}s. Top 5:")
    for r in results[:5]:
        print(f"    proxy={r['proxy']:.4f} | conf={r['conf_thresh']} nms={r['nms_iou']}/{r['nms_type']} "
              f"M={r['M']} α={r['alpha']} T={r['temp']}")
    print(f"\n  Full eval on top 50...")
    t1 = time.time()
    for i, r in enumerate(results[:50]):
        cfg = {k: r[k] for k in ['conf_thresh', 'nms_iou', 'nms_type', 'M', 'alpha', 'temp', 'score_mode']}
        preds, _ = generate_predictions(det_data, cls_data, val_ids, cfg)
        dm, cm, bl = full_eval(preds, gt_coco_data, val_ids)
        r['det_map50'] = dm
        r['cls_map50'] = cm
        r['blend'] = bl
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/50] {time.time()-t1:.0f}s")
    top50 = sorted([r for r in results[:50] if 'blend' in r], key=lambda x: x['blend'], reverse=True)
    print(f"\n  TOP 10 (OOF blend):")
    print(f"  {'#':>3} {'Blend':>7} {'Det':>7} {'Cls':>7} | {'Conf':>6} {'NMS':>5} {'Type':>6} {'M':>4} {'α':>5} {'T':>4}")
    for i, r in enumerate(top50[:10]):
        print(f"  {i+1:3d} {r['blend']:.4f} {r['det_map50']:.4f} {r['cls_map50']:.4f} | "
              f"{r['conf_thresh']:6.3f} {r['nms_iou']:.2f} {r['nms_type']:>6} {r['M']:4d} {r['alpha']:.2f} {r['temp']:.1f}")
    with open(str(output_path), 'w') as f:
        json.dump({'stage_a': top50}, f, indent=2)
    return top50[:5]


def sweep_stage_b(det_data, cls_data, gt_by_image, gt_coco_data, val_ids,
                  ref_emb, ref_labels, ref_counts, base_configs, output_path):
    print("\n" + "=" * 60)
    print("STAGE B SWEEP: kNN routing")
    print("=" * 60)
    rts = [0.3, 0.4, 0.5, 0.6]
    kas = [0.3, 0.4, 0.5, 0.6]
    kts = [0.10, 0.15, 0.20, 0.30]
    rps = [1.0, 0.95, 0.90]
    n_per = len(rts) * len(kas) * len(kts) * len(rps)
    print(f"  {len(base_configs)} bases × {n_per} = {len(base_configs)*n_per} configs")
    all_res = []
    t0 = time.time()
    for bi, base in enumerate(base_configs):
        print(f"\n  Base {bi+1}: blend={base.get('blend','?'):.4f} conf={base['conf_thresh']} "
              f"nms={base['nms_iou']}/{base['nms_type']} M={base['M']} α={base['alpha']} T={base['temp']}")
        for rt in rts:
            for ka in kas:
                for kt in kts:
                    for rp in rps:
                        cfg = {k: base[k] for k in ['conf_thresh', 'nms_iou', 'nms_type', 'M', 'alpha', 'temp']}
                        cfg.update({'route_thresh': rt, 'knn_alpha': ka, 'knn_temp': kt, 'route_penalty': rp})
                        _, pbi = generate_predictions_knn(det_data, cls_data, val_ids, cfg,
                                                           ref_emb, ref_labels, ref_counts)
                        px, dr, ca = proxy_eval(pbi, gt_by_image, val_ids)
                        all_res.append({**cfg, 'proxy': px, 'det_recall': dr, 'cls_acc': ca})
        print(f"    {(bi+1)*n_per} done, {time.time()-t0:.0f}s")
    all_res.sort(key=lambda x: x['proxy'], reverse=True)
    print(f"\n  Full eval on top 20...")
    for i, r in enumerate(all_res[:20]):
        cfg = {k: r[k] for k in r if k not in ['proxy', 'det_recall', 'cls_acc']}
        preds, _ = generate_predictions_knn(det_data, cls_data, val_ids, cfg,
                                             ref_emb, ref_labels, ref_counts)
        dm, cm, bl = full_eval(preds, gt_coco_data, val_ids)
        r['det_map50'] = dm
        r['cls_map50'] = cm
        r['blend'] = bl
    top20 = sorted([r for r in all_res[:20] if 'blend' in r], key=lambda x: x['blend'], reverse=True)
    print(f"\n  TOP 10 kNN:")
    print(f"  {'#':>3} {'Blend':>7} {'Det':>7} {'Cls':>7} | {'RT':>4} {'KA':>4} {'KT':>5} {'RP':>4}")
    for i, r in enumerate(top20[:10]):
        print(f"  {i+1:3d} {r['blend']:.4f} {r['det_map50']:.4f} {r['cls_map50']:.4f} | "
              f"{r['route_thresh']:.1f} {r['knn_alpha']:.1f} {r['knn_temp']:.2f} {r['route_penalty']:.2f}")
    existing = {}
    if output_path.exists():
        with open(str(output_path)) as f:
            existing = json.load(f)
    existing['stage_b'] = top20
    with open(str(output_path), 'w') as f:
        json.dump(existing, f, indent=2)
    return top20[:5]


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/root/ng/data')
    parser.add_argument('--cache_dir', default='/root/ng/cache')
    parser.add_argument('--weights', default='')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--skip_classify', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_path = Path(args.data) / 'train' / 'annotations.json'
    img_dir = Path(args.data) / 'train' / 'images'
    det_cache_dir = Path(args.cache_dir) / 'detections'
    output_path = Path(args.cache_dir) / f'sweep_{args.tag}.json'

    with open(ann_path) as f:
        coco_data = json.load(f)
    id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}

    if args.fold > 0:
        train_ids, val_ids = get_fold_split(str(ann_path), args.fold)
        print(f"Fold {args.fold}: {len(train_ids)} train, {len(val_ids)} val")
    else:
        all_ids = sorted(set(img['id'] for img in coco_data['images']))
        train_ids = val_ids = all_ids
        print(f"All {len(all_ids)} images (contaminated)")

    gt_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    # Load cached detections
    print("\nLoading cached detections...")
    det_data = {}
    for img_id in val_ids:
        p = det_cache_dir / f'{img_id}.npz'
        if p.exists():
            det_data[img_id] = dict(np.load(str(p)))
    print(f"  {len(det_data)} images loaded")

    # Classify or load cache
    cls_cache_dir = Path(args.cache_dir) / f'cls_{args.tag}'
    cls_cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_classify:
        assert args.weights, "--weights required unless --skip_classify"
        print(f"\nClassifier: {args.weights}")
        model = load_classifier(args.weights, device)
        print("Classifying...")
        cls_data = classify_cached_boxes(model, img_dir, id_to_file, det_cache_dir, val_ids, device, args.batch_size)
        for img_id, cd in cls_data.items():
            np.savez_compressed(str(cls_cache_dir / f'{img_id}.npz'), **cd)
        print("\nBuilding ref embeddings...")
        ref_emb, ref_labels, ref_counts = build_ref_embeddings(
            model, str(ann_path), img_dir, train_ids, args.data, device, args.batch_size)
        np.savez_compressed(str(cls_cache_dir / 'ref.npz'),
                            ref_emb=ref_emb, ref_labels=ref_labels, ref_counts=ref_counts)
        del model
        torch.cuda.empty_cache()
    else:
        print("\nLoading cached classifications...")
        cls_data = {}
        for img_id in val_ids:
            p = cls_cache_dir / f'{img_id}.npz'
            if p.exists():
                cls_data[img_id] = dict(np.load(str(p)))
        ref_cache = np.load(str(cls_cache_dir / 'ref.npz'))
        ref_emb = ref_cache['ref_emb']
        ref_labels = ref_cache['ref_labels']
        ref_counts = ref_cache['ref_counts']
        print(f"  {len(cls_data)} images, {ref_emb.shape[0]} ref embeddings")

    # Ceilings
    run_ceilings(det_data, cls_data, gt_by_image, val_ids, coco_data)

    # Baselines
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)

    configs = [
        ("v4.1 (fused 0.7/0.3)",
         {'conf_thresh': 0.001, 'nms_iou': 0.5, 'nms_type': 'hard', 'M': 200, 'alpha': 0.3, 'temp': 1.0, 'score_mode': 'add'}),
        ("v4.4 (pure det_score)",
         {'conf_thresh': 0.001, 'nms_iou': 0.5, 'nms_type': 'hard', 'M': 300, 'alpha': 0.0, 'temp': 1.0, 'score_mode': 'det'}),
        ("det*cls^0.15 (CTO rec)",
         {'conf_thresh': 0.001, 'nms_iou': 0.5, 'nms_type': 'hard', 'M': 300, 'alpha': 0.15, 'temp': 1.0, 'score_mode': 'mul'}),
        ("det*cls^0.25",
         {'conf_thresh': 0.001, 'nms_iou': 0.5, 'nms_type': 'hard', 'M': 300, 'alpha': 0.25, 'temp': 1.0, 'score_mode': 'mul'}),
    ]
    for name, cfg in configs:
        preds, _ = generate_predictions(det_data, cls_data, val_ids, cfg)
        dm, cm, bl = full_eval(preds, coco_data, val_ids)
        print(f"  {name:30s}: det={dm:.4f} cls={cm:.4f} blend={bl:.4f}")

    # v4.4 + kNN baseline
    knn_cfg = {'conf_thresh': 0.001, 'nms_iou': 0.5, 'nms_type': 'hard', 'M': 300,
               'alpha': 0.0, 'temp': 1.0, 'route_thresh': 0.5, 'knn_alpha': 0.45,
               'knn_temp': 0.15, 'route_penalty': 1.0}
    preds, _ = generate_predictions_knn(det_data, cls_data, val_ids, knn_cfg,
                                         ref_emb, ref_labels, ref_counts)
    dm, cm, bl = full_eval(preds, coco_data, val_ids)
    print(f"  {'v4.4 + kNN':30s}: det={dm:.4f} cls={cm:.4f} blend={bl:.4f}")

    # Sweep
    best_a = sweep_stage_a(det_data, cls_data, gt_by_image, coco_data, val_ids, output_path)
    best_b = sweep_stage_b(det_data, cls_data, gt_by_image, coco_data, val_ids,
                           ref_emb, ref_labels, ref_counts, best_a, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("DONE — results saved to", output_path)
    print("=" * 60)


if __name__ == '__main__':
    main()
