"""
Fine-tune EVA-02 base as 356-class supervised classifier.
Trains on COCO training crops + product reference images.
Replaces kNN with learned classification head → expected +0.10-0.15 cls mAP.

Usage: python train_classifier.py
Output: /clade/ng/submission/classifier_supervised.safetensors
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
import timm
from safetensors.torch import save_file
from pathlib import Path
from PIL import Image
import json
import time
import csv

# ─── Config ───────────────────────────────────────────────────────
DATA_ROOT = Path("/mnt/SFS-qZE4t9Aw/data")
COCO_ROOT = DATA_ROOT / "coco_dataset" / "train"
PRODUCT_ROOT = DATA_ROOT / "product_images"
YOLO_ROOT = DATA_ROOT / "yolo"
OUTPUT_DIR = Path("/clade/ng/submission")
LOG_DIR = Path("/clade/ng")

MODEL_NAME = "eva02_base_patch16_clip_224"
NUM_CLASSES = 356
BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 2e-4
BACKBONE_LR_SCALE = 0.1
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
LABEL_SMOOTHING = 0.1
DEVICE = "cuda:0"


# ─── Datasets ─────────────────────────────────────────────────────

class CropDataset(Dataset):
    """COCO annotation crops with on-the-fly cropping."""

    def __init__(self, annotations, images_dir, image_info, transform):
        self.anns = annotations
        self.img_dir = images_dir
        self.img_info = image_info
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        iid = ann["image_id"]
        if iid not in self._cache:
            info = self.img_info[iid]
            self._cache[iid] = Image.open(
                str(self.img_dir / info["file_name"])
            ).convert("RGB")
            if len(self._cache) > 30:
                self._cache.pop(next(iter(self._cache)))
        img = self._cache[iid]
        iw, ih = img.size
        x, y, w, h = ann["bbox"]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(iw, int(x + w))
        y2 = min(ih, int(y + h))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, min(32, iw), min(32, ih)
        crop = img.crop((x1, y1, x2, y2))
        return self.transform(crop), ann["category_id"]


class ProductDataset(Dataset):
    """Product reference images."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(str(path)).convert("RGB")
        return self.transform(img), label


# ─── Main ─────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"{'=' * 60}")
    print(f"Fine-tuning {MODEL_NAME} → {NUM_CLASSES}-class classifier")
    print(f"Device: {DEVICE} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"LR: {LR} (backbone: {LR * BACKBONE_LR_SCALE}) | WD: {WEIGHT_DECAY}")
    print(f"{'=' * 60}", flush=True)

    # Load COCO annotations
    with open(COCO_ROOT / "annotations.json") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}

    # Verify category IDs are 0..355
    all_cat_ids = sorted(c["id"] for c in coco["categories"])
    assert all_cat_ids == list(range(NUM_CLASSES)), \
        f"Expected cat IDs 0..{NUM_CLASSES-1}, got {all_cat_ids[:5]}..{all_cat_ids[-5:]}"

    # Train/val split from YOLO directories
    train_stems = set(
        p.stem for p in (YOLO_ROOT / "train" / "images").iterdir()
    )
    val_stems = set(
        p.stem for p in (YOLO_ROOT / "val" / "images").iterdir()
    )
    train_ids, val_ids = set(), set()
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        if stem in train_stems:
            train_ids.add(im["id"])
        elif stem in val_stems:
            val_ids.add(im["id"])

    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    val_anns = [a for a in coco["annotations"] if a["image_id"] in val_ids]
    print(f"Train crops: {len(train_anns)} ({len(train_ids)} images)")
    print(f"Val crops:   {len(val_anns)} ({len(val_ids)} images)", flush=True)

    # Product reference images
    with open(PRODUCT_ROOT / "metadata.json") as f:
        meta = json.load(f)
    code_to_cat = {}
    for prod in meta["products"]:
        if prod["product_name"] in cat_name_to_id:
            code_to_cat[prod["product_code"]] = cat_name_to_id[prod["product_name"]]

    product_items = []
    for folder in sorted(PRODUCT_ROOT.iterdir()):
        if folder.is_dir() and folder.name in code_to_cat:
            cat_id = code_to_cat[folder.name]
            for p in sorted(folder.glob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    product_items.append((p, cat_id))
    print(f"Product ref images: {len(product_items)}", flush=True)

    # Transforms (matching EVA-02 CLIP defaults)
    train_tf = timm.data.create_transform(
        input_size=224,
        is_training=True,
        auto_augment="rand-m9-mstd0.5",
        re_prob=0.25,
        re_mode="pixel",
        interpolation="bicubic",
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    val_tf = timm.data.create_transform(
        input_size=224,
        is_training=False,
        interpolation="bicubic",
        crop_pct=224 / 256,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    # Build datasets
    train_ds = ConcatDataset([
        CropDataset(train_anns, COCO_ROOT / "images", images, train_tf),
        ProductDataset(product_items, train_tf),
    ])
    val_ds = CropDataset(val_anns, COCO_ROOT / "images", images, val_tf)

    # Weighted sampler for class balance
    all_labels = [a["category_id"] for a in train_anns] + [p[1] for p in product_items]
    counts = {}
    for l in all_labels:
        counts[l] = counts.get(l, 0) + 1
    sample_weights = [1.0 / counts[l] for l in all_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    print(f"Train steps/epoch: {len(train_loader)} | Val steps: {len(val_loader)}", flush=True)

    # Model: EVA-02 base + 356-class head
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    n_bb = sum(p.numel() for n, p in model.named_parameters() if "head" not in n)
    n_hd = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
    print(f"Params: backbone {n_bb / 1e6:.1f}M + head {n_hd / 1e3:.1f}K", flush=True)

    # Optimizer with differential LR
    bb_params = [p for n, p in model.named_parameters() if "head" not in n]
    hd_params = [p for n, p in model.named_parameters() if "head" in n]
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": LR * BACKBONE_LR_SCALE},
        {"params": hd_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    # LR schedule: linear warmup → cosine decay
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler()

    # CSV log
    log_path = LOG_DIR / "train_classifier.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc", "val_acc", "val_top5", "time_s"
        ])

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        # ── Train ──────────────────────────────────────────────
        model.train()
        rl, rc, rt = 0.0, 0, 0
        te = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sched.step()

            rl += loss.item()
            rc += (logits.argmax(1) == labels).sum().item()
            rt += labels.size(0)

        train_loss = rl / len(train_loader)
        train_acc = rc / rt

        # ── Validate ───────────────────────────────────────────
        model.eval()
        vc, vt, v5 = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                vc += (logits.argmax(1) == labels).sum().item()
                _, top5 = logits.topk(5, dim=1)
                v5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
                vt += labels.size(0)

        val_acc = vc / vt if vt else 0
        val_top5 = v5 / vt if vt else 0
        elapsed = time.time() - te

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            sd = {k: v.half() for k, v in model.state_dict().items()}
            save_file(sd, str(OUTPUT_DIR / "classifier_supervised.safetensors"))
            marker = " ★"

        print(
            f"E{epoch + 1:02d} | loss {train_loss:.4f} | "
            f"train {train_acc:.3f} | val {val_acc:.3f} top5 {val_top5:.3f} | "
            f"{elapsed:.1f}s{marker}",
            flush=True,
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_acc:.4f}",
                f"{val_top5:.4f}",
                f"{elapsed:.1f}",
            ])

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {total:.0f}s")
    print(f"Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    sz = (OUTPUT_DIR / "classifier_supervised.safetensors").stat().st_size / 1e6
    print(f"Model: {sz:.1f} MB")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
