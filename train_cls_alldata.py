"""
Fine-tune EVA-02 on ALL 248 images + product refs (no val holdout).
Saves checkpoints every 5 epochs + final. Best model = last epoch.

Usage: python train_cls_alldata.py
Output: /clade/ng/submission/classifier.safetensors
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
OUTPUT_DIR = Path("/clade/ng/submission")
LOG_DIR = Path("/clade/ng")

MODEL_NAME = "eva02_base_patch16_clip_224"
NUM_CLASSES = 356
BATCH_SIZE = 256
NUM_EPOCHS = 40
LR = 2e-4
BACKBONE_LR_SCALE = 0.1
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
LABEL_SMOOTHING = 0.1
DEVICE = "cuda:0"


class CropDataset(Dataset):
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
            if len(self._cache) > 50:
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
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(str(path)).convert("RGB")
        return self.transform(img), label


def main():
    t0 = time.time()
    print(f"{'=' * 60}")
    print(f"ALL-DATA training: {MODEL_NAME} → {NUM_CLASSES}-class")
    print(f"Device: {DEVICE} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"{'=' * 60}", flush=True)

    with open(COCO_ROOT / "annotations.json") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}

    all_cat_ids = sorted(c["id"] for c in coco["categories"])
    assert all_cat_ids == list(range(NUM_CLASSES))

    # ALL annotations (no split)
    all_anns = coco["annotations"]
    all_ids = set(im["id"] for im in coco["images"])
    print(f"All crops: {len(all_anns)} ({len(all_ids)} images)")

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

    train_ds = ConcatDataset([
        CropDataset(all_anns, COCO_ROOT / "images", images, train_tf),
        ProductDataset(product_items, train_tf),
    ])

    all_labels = [a["category_id"] for a in all_anns] + [p[1] for p in product_items]
    counts = {}
    for l in all_labels:
        counts[l] = counts.get(l, 0) + 1
    sample_weights = [1.0 / counts[l] for l in all_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    print(f"Train steps/epoch: {len(train_loader)}", flush=True)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    n_bb = sum(p.numel() for n, p in model.named_parameters() if "head" not in n)
    n_hd = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
    print(f"Params: backbone {n_bb / 1e6:.1f}M + head {n_hd / 1e3:.1f}K", flush=True)

    bb_params = [p for n, p in model.named_parameters() if "head" not in n]
    hd_params = [p for n, p in model.named_parameters() if "head" in n]
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": LR * BACKBONE_LR_SCALE},
        {"params": hd_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

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

    log_path = LOG_DIR / "train_cls_alldata.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "time_s"])

    for epoch in range(NUM_EPOCHS):
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
        elapsed = time.time() - te

        # Save every epoch (spot instance — may be evicted)
        saved = ""
        if True:
            sd = {k: v.half() for k, v in model.state_dict().items()}
            save_file(sd, str(OUTPUT_DIR / "classifier.safetensors"))
            saved = " [SAVED]"

        print(
            f"E{epoch + 1:02d} | loss {train_loss:.4f} | "
            f"train {train_acc:.3f} | {elapsed:.1f}s{saved}",
            flush=True,
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{elapsed:.1f}",
            ])

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {total:.0f}s")
    sz = (OUTPUT_DIR / "classifier.safetensors").stat().st_size / 1e6
    print(f"Model: {sz:.1f} MB at {OUTPUT_DIR / 'classifier.safetensors'}")


if __name__ == "__main__":
    main()
