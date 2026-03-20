"""Train improved U-Net v3 with z-augmentation and larger model.

Key improvements over v2:
1. Z-augmentation: create synthetic catastrophic/healthy examples by interpolating GT
2. Larger model (256 hidden, 8 ResBlocks)
3. Multi-scale features (add 2x downsampled branch)
4. Mixup augmentation
5. Longer training with warmup

Run on A100: python train_nn_v3.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

GT_DIR = Path("ground_truth")
CAL_FILE = Path("calibration.json")
NUM_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
IN_CHANNELS = 13


def encode_grid(grid: np.ndarray, z: float) -> np.ndarray:
    h, w = grid.shape
    channels = []
    for code in TERRAIN_CODES:
        channels.append((grid == code).astype(np.float32))
    positions = list(zip(*np.where(np.isin(grid, [1, 2]))))
    dist = np.full((h, w), 40.0, dtype=np.float32)
    if positions:
        for sy, sx in positions:
            yy = np.abs(np.arange(h) - sy)[:, None].astype(np.float32)
            xx = np.abs(np.arange(w) - sx)[None, :].astype(np.float32)
            dist = np.minimum(dist, yy + xx)
    channels.append(dist / 40.0)
    ocean = (grid == 10)
    coastal = np.zeros((h, w), dtype=np.float32)
    if h > 1:
        coastal[1:] = np.maximum(coastal[1:], ocean[:-1])
        coastal[:-1] = np.maximum(coastal[:-1], ocean[1:])
    if w > 1:
        coastal[:, 1:] = np.maximum(coastal[:, 1:], ocean[:, :-1])
        coastal[:, :-1] = np.maximum(coastal[:, :-1], ocean[:, 1:])
    channels.append(coastal)
    channels.append((grid != 10).astype(np.float32))
    channels.append(np.full((h, w), z, dtype=np.float32))
    settle = np.isin(grid, [1, 2]).astype(np.float32)
    padded = np.pad(settle, 2, mode="constant")
    density = np.zeros((h, w), dtype=np.float32)
    for dy in range(5):
        for dx in range(5):
            density += padded[dy : dy + h, dx : dx + w]
    channels.append(density / 25.0)
    return np.stack(channels)


def load_dataset(gt_dir, cal_file):
    round_z = {}
    if cal_file.exists():
        cal = json.loads(cal_file.read_text())
        round_z = {int(k): v for k, v in cal.get("round_z", {}).items()}
    samples = []
    for gt_file in sorted(Path(gt_dir).glob("round_*_seed_*.json")):
        data = json.loads(gt_file.read_text())
        rn = int(gt_file.stem.split("_")[1])
        grid = np.array(data["initial_grid"], dtype=np.int32)
        gt = np.array(data["ground_truth"], dtype=np.float32)
        z = round_z.get(rn, 0.283)
        samples.append((grid, gt, z, rn))
    return samples


# ── Z-Augmentation ───────────────────────────────────────────────────


def z_augment_sample(grid, gt, z, target_z, rng):
    """Create synthetic sample at different z by interpolating GT.

    For catastrophic z (low): push settlements/ports toward ruin/empty
    For healthy z (high): push ruins toward settlements
    """
    if abs(target_z - z) < 0.05:
        return gt.copy()

    h, w, c = gt.shape
    new_gt = gt.copy()

    # Interpolation factor: how much to shift
    shift = (target_z - z)  # positive = more healthy, negative = more catastrophic

    for y in range(h):
        for x in range(w):
            # Only modify dynamic cells (not ocean/mountain)
            if grid[y, x] in (10, 5):
                continue

            p = new_gt[y, x].copy()

            if shift < 0:
                # More catastrophic: settlements → ruins → empty
                transfer = min(p[1], -shift * 0.5)  # settlement prob
                p[1] -= transfer
                p[3] += transfer * 0.3  # some to ruin
                p[0] += transfer * 0.7  # some to empty

                transfer_p = min(p[2], -shift * 0.5)  # port prob
                p[2] -= transfer_p
                p[3] += transfer_p * 0.3
                p[0] += transfer_p * 0.7

                # Ruins → forest/empty
                transfer_r = min(p[3], -shift * 0.3)
                p[3] -= transfer_r
                p[4] += transfer_r * 0.5  # forest
                p[0] += transfer_r * 0.5  # empty
            else:
                # More healthy: empty/ruins → settlements
                transfer = min(p[0] * 0.3, shift * 0.3)
                p[0] -= transfer
                if grid[y, x] in (1, 2, 11):
                    p[1] += transfer * 0.7
                    p[2] += transfer * 0.3
                else:
                    p[4] += transfer

                transfer_r = min(p[3], shift * 0.3)
                p[3] -= transfer_r
                p[1] += transfer_r * 0.5
                p[0] += transfer_r * 0.5

            # Ensure valid distribution
            p = np.clip(p, 0.001, None)
            new_gt[y, x] = p / p.sum()

    return new_gt


# ── Model ─────────────────────────────────────────────────────────────


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout2d(0.05)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(self.bn2(self.conv2(h)))
        return F.relu(x + h)


class AstarNetV3(nn.Module):
    """Larger dilated ResNet with multi-scale features."""

    def __init__(self, in_ch=IN_CHANNELS, hidden=192, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        # Main branch: dilated convolutions
        self.blocks = nn.Sequential(
            ResBlock(hidden, dilation=1),
            ResBlock(hidden, dilation=2),
            ResBlock(hidden, dilation=4),
            ResBlock(hidden, dilation=8),
            ResBlock(hidden, dilation=16),
            ResBlock(hidden, dilation=8),
            ResBlock(hidden, dilation=4),
            ResBlock(hidden, dilation=1),
        )
        # Downsampled branch: captures global context
        self.down_conv = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        # Merge and output
        self.merge = nn.Sequential(
            nn.Conv2d(hidden + hidden // 2, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        main = self.blocks(h)
        down = self.down_conv(h)
        merged = self.merge(torch.cat([main, down], dim=1))
        logits = self.head(merged)
        return logits.permute(0, 2, 3, 1)


# ── Loss ──────────────────────────────────────────────────────────────


def competition_loss(pred_logits, target, eps=1e-8):
    pred_prob = F.softmax(pred_logits, dim=-1)
    pred_prob = torch.clamp(pred_prob, min=0.01)
    pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(target * torch.log(torch.clamp(target, min=eps)), dim=-1)
    kl = torch.sum(
        target * torch.log(torch.clamp(target, min=eps) / (pred_prob + eps)), dim=-1
    )
    total_entropy = entropy.sum(dim=(-2, -1), keepdim=True) + eps
    weight = entropy / total_entropy
    weighted_kl = (weight * kl).sum(dim=(-2, -1))
    return weighted_kl.mean()


def compute_score(pred_prob_np, gt_np):
    eps = 1e-10
    entropy = -np.sum(gt_np * np.log(np.clip(gt_np, eps, 1)), axis=-1)
    kl = np.sum(
        gt_np * np.log(np.clip(gt_np, eps, 1) / np.clip(pred_prob_np, eps, 1)),
        axis=-1,
    )
    w = entropy / (entropy.sum() + eps)
    wkl = (w * kl).sum()
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


# ── Training ──────────────────────────────────────────────────────────


def make_batch(samples, rng, augment=True, z_augment=True):
    """Create batch with rotation + z augmentation."""
    features, targets = [], []
    for grid, gt, z, rn in samples:
        g, t = grid.copy(), gt.copy()

        # Rotation/flip augmentation
        if augment:
            aug = rng.integers(8)
            rot = aug % 4
            flip = aug >= 4
            g = np.rot90(g, rot).copy()
            t = np.rot90(t, rot, axes=(0, 1)).copy()
            if flip:
                g = np.fliplr(g).copy()
                t = np.fliplr(t).copy()

        # Z augmentation: sometimes shift z and adjust GT
        z_used = z
        if z_augment and augment and rng.random() < 0.3:
            # Random target z
            target_z = rng.uniform(0.01, 0.45)
            t = z_augment_sample(g, t, z, target_z, rng)
            z_used = target_z

        features.append(encode_grid(g, z_used))
        targets.append(t)

    return (
        torch.tensor(np.stack(features), dtype=torch.float32).to(DEVICE),
        torch.tensor(np.stack(targets), dtype=torch.float32).to(DEVICE),
    )


def train_model(samples, epochs=1500, lr=3e-4, tag="full"):
    model = AstarNetV3().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    # Warmup + cosine decay
    warmup_epochs = 50
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] Model: {n_params:,} params, {len(samples)} samples, {DEVICE}")

    rng = np.random.default_rng(42)
    best_score = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(samples))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(idx), 8):
            batch_samples = [samples[i] for i in idx[start : start + 8]]
            features, targets = make_batch(batch_samples, rng, augment=True, z_augment=True)

            pred = model(features)
            loss = competition_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            scores = []
            with torch.no_grad():
                features, targets = make_batch(samples, rng, augment=False, z_augment=False)
                pred = model(features)
                pred_prob = F.softmax(pred, dim=-1)
                pred_prob = torch.clamp(pred_prob, min=0.01)
                pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)
                for i in range(len(samples)):
                    s = compute_score(
                        pred_prob[i].cpu().numpy(), targets[i].cpu().numpy()
                    )
                    scores.append(s)

            avg = np.mean(scores)
            mn = np.min(scores)
            print(
                f"  [{tag}] Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, "
                f"avg={avg:.1f}, min={mn:.1f}, lr={scheduler.get_last_lr()[0]:.6f}"
            )

            if avg > best_score:
                best_score = avg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_score


def loro_cv(samples, epochs=800):
    rounds = sorted(set(s[3] for s in samples))
    print(f"\n{'='*60}")
    print(f"LORO Cross-Validation ({len(rounds)} rounds)")
    print(f"{'='*60}")

    loro_scores = {}
    for test_round in rounds:
        train_s = [s for s in samples if s[3] != test_round]
        test_s = [s for s in samples if s[3] == test_round]

        best_state, train_score = train_model(
            train_s, epochs=epochs, tag=f"LORO-R{test_round}"
        )

        model = AstarNetV3().to(DEVICE)
        model.load_state_dict(best_state)
        model.eval()

        rng = np.random.default_rng(0)
        round_scores = []
        with torch.no_grad():
            features, targets = make_batch(test_s, rng, augment=False, z_augment=False)
            pred = model(features)
            pred_prob = F.softmax(pred, dim=-1)
            pred_prob = torch.clamp(pred_prob, min=0.01)
            pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)
            for i in range(len(test_s)):
                s = compute_score(
                    pred_prob[i].cpu().numpy(), targets[i].cpu().numpy()
                )
                round_scores.append(s)

        avg = np.mean(round_scores)
        loro_scores[test_round] = avg
        print(f"  Round {test_round} holdout: {avg:.1f} (train fit: {train_score:.1f})")

    overall = np.mean(list(loro_scores.values()))
    print(f"\nLORO average: {overall:.1f}")
    print(f"Per-round: {loro_scores}")
    return loro_scores


def main():
    print(f"Loading GT data from {GT_DIR}...")
    samples = load_dataset(GT_DIR, CAL_FILE)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("No GT files found!")
        return

    # Phase 1: Full training
    print(f"\n{'='*60}")
    print("Phase 1: Full Training v3 (z-augmented, larger model)")
    print(f"{'='*60}")
    best_state, best_score = train_model(samples, epochs=1500, tag="FULL")
    print(f"\nBest full-data score: {best_score:.1f}")
    torch.save(best_state, "astar_nn_v3.pt")
    print("Model saved to astar_nn_v3.pt")

    # Phase 2: LORO
    loro_scores = loro_cv(samples, epochs=800)

    print("\nDone!")


if __name__ == "__main__":
    main()
