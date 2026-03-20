"""
ArcFace + CE classifier training for NM i AI 2026 NorgesGruppen task.
Train EVA-02 Base with dual loss: CrossEntropy + ArcFace margin.
Balanced sampler oversamples rare classes.

Usage:
  python train_arcface.py --data /root/ng/data --output /root/ng/output --epochs 30
  python train_arcface.py --data /root/ng/data --output /root/ng/output --epochs 30 --fold 1  # OOF eval
"""
import argparse
import json
import math
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import timm

NUM_CLASSES = 356
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ArcFaceHead(nn.Module):
    """Additive Angular Margin Loss (ArcFace)."""
    def __init__(self, in_features, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # L2 normalize
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        # Add angular margin to target class
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta[torch.arange(len(labels)), labels] + self.margin)
        one_hot = F.one_hot(labels, NUM_CLASSES).float()
        output = cosine * (1 - one_hot) + target_logits.unsqueeze(1) * one_hot
        return output * self.scale


class CropDataset(Dataset):
    """Crops from COCO annotations for classification training."""
    def __init__(self, annotations_path, image_dir, image_ids=None, augment=True):
        with open(annotations_path) as f:
            coco = json.load(f)

        self.image_dir = Path(image_dir)
        self.augment = augment

        # Build image lookup
        id_to_file = {img['id']: img['file_name'] for img in coco['images']}

        # Filter by image_ids if provided (for fold splits)
        if image_ids is not None:
            valid_ids = set(image_ids)
        else:
            valid_ids = set(id_to_file.keys())

        self.samples = []
        for ann in coco['annotations']:
            if ann['image_id'] not in valid_ids:
                continue
            x, y, w, h = ann['bbox']
            if w < 2 or h < 2:
                continue
            self.samples.append({
                'image_file': str(self.image_dir / id_to_file[ann['image_id']]),
                'bbox': [x, y, w, h],
                'category_id': ann['category_id'],
            })

        # Category distribution
        self.cat_counts = Counter(s['category_id'] for s in self.samples)
        print(f"Dataset: {len(self.samples)} crops, {len(self.cat_counts)} classes")
        rare = sum(1 for c, n in self.cat_counts.items() if n <= 5)
        print(f"Rare classes (<=5): {rare}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image_file']).convert('RGB')
        x, y, w, h = s['bbox']
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))

        # Resize(256) -> CenterCrop(224) — match timm EVA-02 eval transform
        cw, ch = crop.size
        scale = 256 / min(cw, ch)
        nw, nh = round(cw * scale), round(ch * scale)
        crop = crop.resize((nw, nh), Image.BICUBIC)

        if self.augment:
            # Random crop instead of center crop
            left = torch.randint(0, max(1, nw - 224), (1,)).item()
            top = torch.randint(0, max(1, nh - 224), (1,)).item()
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            left = (nw - 224) // 2
            top = (nh - 224) // 2

        crop = crop.crop((left, top, left + 224, top + 224))
        arr = np.array(crop, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

        # Normalize
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, s['category_id']


class ProductImageDataset(Dataset):
    """Product reference images as additional training data."""
    def __init__(self, product_dir, ean_map_path, augment=True):
        self.product_dir = Path(product_dir)
        self.augment = augment
        self.samples = []

        # Load EAN code -> category_id mapping
        with open(ean_map_path) as f:
            ean_to_catid = json.load(f)

        for cat_dir in sorted(self.product_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_id = ean_to_catid.get(cat_dir.name)
            if cat_id is None:
                continue
            for img_path in cat_dir.iterdir():
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append({
                        'image_file': str(img_path),
                        'category_id': cat_id,
                    })

        self.cat_counts = Counter(s['category_id'] for s in self.samples)
        print(f"Product images: {len(self.samples)} across {len(self.cat_counts)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image_file']).convert('RGB')

        # Resize to 256, then crop to 224
        w, h = img.size
        scale = 256 / min(w, h)
        nw, nh = round(w * scale), round(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)

        if self.augment:
            left = torch.randint(0, max(1, nw - 224), (1,)).item()
            top = torch.randint(0, max(1, nh - 224), (1,)).item()
            if torch.rand(1).item() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            left = (nw - 224) // 2
            top = (nh - 224) // 2

        img = img.crop((left, top, left + 224, top + 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, s['category_id']


def make_balanced_sampler(dataset):
    """Weighted sampler: oversample rare classes to min 20 effective examples."""
    cat_counts = dataset.cat_counts
    min_target = 20
    weights = []
    for s in dataset.samples:
        c = s['category_id']
        count = cat_counts[c]
        # Oversample factor: rare classes get higher weight
        w = max(1.0, min_target / count)
        weights.append(w)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_fold_split(annotations_path, fold=1, n_folds=5, seed=42):
    """Split image IDs into train/val for OOF evaluation."""
    with open(annotations_path) as f:
        coco = json.load(f)
    image_ids = sorted([img['id'] for img in coco['images']])
    rng = np.random.RandomState(seed)
    rng.shuffle(image_ids)
    fold_size = len(image_ids) // n_folds
    val_start = (fold - 1) * fold_size
    val_end = val_start + fold_size if fold < n_folds else len(image_ids)
    val_ids = image_ids[val_start:val_end]
    train_ids = [i for i in image_ids if i not in set(val_ids)]
    return train_ids, val_ids


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data)
    ann_path = data_dir / 'train' / 'annotations.json'
    img_dir = data_dir / 'train' / 'images'
    product_dir = data_dir  # Product EAN folders are at data root level

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fold split or full training
    if args.fold > 0:
        print(f"\n=== OOF Fold {args.fold}/{args.n_folds} ===")
        train_ids, val_ids = get_fold_split(ann_path, args.fold, args.n_folds)
        print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")
        train_dataset = CropDataset(ann_path, img_dir, image_ids=train_ids, augment=True)
        val_dataset = CropDataset(ann_path, img_dir, image_ids=val_ids, augment=False)
    else:
        print("\n=== Full training (all images) ===")
        train_dataset = CropDataset(ann_path, img_dir, image_ids=None, augment=True)
        val_dataset = None

    # Add product images to training
    ean_map_path = data_dir / 'ean_to_catid.json'
    if product_dir.exists() and args.use_products and ean_map_path.exists():
        prod_dataset = ProductImageDataset(product_dir, str(ean_map_path), augment=True)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, prod_dataset])
        # Rebuild cat_counts for sampler
        all_counts = Counter()
        for ds in [train_dataset.datasets[0], prod_dataset]:
            all_counts.update(ds.cat_counts)
        train_dataset.cat_counts = all_counts
        train_dataset.samples = (
            train_dataset.datasets[0].samples + prod_dataset.samples
        )

    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

    # Model: EVA-02 Base with dual heads
    model = timm.create_model(
        'eva02_base_patch16_clip_224', pretrained=True, num_classes=NUM_CLASSES
    )

    # Load existing fine-tuned weights if available
    if args.resume:
        from safetensors.torch import load_file
        sd = load_file(args.resume)
        sd = {k: v.float() for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"Resumed from {args.resume}")

    model = model.to(device)

    # ArcFace head (separate from CE head)
    embed_dim = 768  # EVA-02 Base feature dim
    arcface = ArcFaceHead(embed_dim, NUM_CLASSES, scale=args.arc_scale, margin=args.arc_margin)
    arcface = arcface.to(device)

    # Optimizer
    params = list(model.parameters()) + list(arcface.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    # Cosine LR schedule
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss weights
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_rare_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        arcface.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward: get features and CE logits
            features = model.forward_features(images)
            ce_logits = model.forward_head(features)

            # CLS token for ArcFace
            cls_feat = features[:, 0]  # [B, 768]

            # ArcFace logits
            arc_logits = arcface(cls_feat, labels)

            # Dual loss
            loss_ce = ce_loss_fn(ce_logits, labels)
            loss_arc = ce_loss_fn(arc_logits, labels)
            loss = args.ce_weight * loss_ce + args.arc_weight * loss_arc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = ce_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if (batch_idx + 1) % 50 == 0:
                print(f"  E{epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={total_loss/(batch_idx+1):.4f} acc={correct/total:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} train_acc={train_acc:.4f}")

        # Validation
        if val_dataset:
            model.train(False)
            val_correct = 0
            val_total = 0
            rare_correct = 0
            rare_total = 0

            # Identify rare classes from training set
            train_cats = train_dataset.datasets[0].cat_counts if hasattr(train_dataset, 'datasets') else train_dataset.cat_counts
            rare_classes = {c for c, n in train_cats.items() if n <= 5}

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    features = model.forward_features(images)
                    logits = model.forward_head(features)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += len(labels)
                    # Rare class accuracy
                    for p, l in zip(preds, labels):
                        if l.item() in rare_classes:
                            rare_total += 1
                            if p.item() == l.item():
                                rare_correct += 1

            val_acc = val_correct / val_total if val_total > 0 else 0
            rare_acc = rare_correct / rare_total if rare_total > 0 else 0
            print(f"  Val acc={val_acc:.4f} | Rare acc={rare_acc:.4f} ({rare_correct}/{rare_total})")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_rare_acc = rare_acc
                # Save best
                from safetensors.torch import save_file
                sd = {k: v.half() for k, v in model.state_dict().items()}
                save_file(sd, str(output_dir / 'classifier_arcface_best.safetensors'))
                print(f"  Saved best (val_acc={val_acc:.4f}, rare_acc={rare_acc:.4f})")

    # Always save final
    from safetensors.torch import save_file
    sd = {k: v.half() for k, v in model.state_dict().items()}
    save_file(sd, str(output_dir / 'classifier_arcface.safetensors'))
    print(f"\nDone. Final weights saved.")
    if val_dataset:
        print(f"Best val_acc={best_val_acc:.4f}, rare_acc={best_rare_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data dir')
    parser.add_argument('--output', required=True, help='Output dir for weights')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0=full, 1-5=OOF)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--resume', type=str, default='', help='Path to safetensors to resume from')
    parser.add_argument('--use_products', action='store_true', help='Include product images in training')
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--arc_weight', type=float, default=0.5)
    parser.add_argument('--arc_scale', type=float, default=64.0)
    parser.add_argument('--arc_margin', type=float, default=0.5)
    train(parser.parse_args())
