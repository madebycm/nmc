"""Train U-Net to predict end-state probabilities from initial grid + z.

25 GT files (R1-R5 × 5 seeds), augmented 8× with rotations/flips.
Loss: entropy-weighted KL divergence (matches competition scoring exactly).
Includes LORO cross-validation to measure generalization.

Run on H100: python train_nn.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Compat patches
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

GT_DIR = Path("ground_truth")
CAL_FILE = Path("calibration.json")
NUM_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Data ──────────────────────────────────────────────────────────────

TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]  # 8 channels


def encode_grid(grid: np.ndarray, z: float) -> np.ndarray:
    """Encode 40×40 grid → (C, 40, 40) feature tensor."""
    h, w = grid.shape
    channels = []

    # One-hot terrain (8 ch)
    for code in TERRAIN_CODES:
        channels.append((grid == code).astype(np.float32))

    # Distance to nearest settlement/port (1 ch, normalized)
    positions = list(zip(*np.where(np.isin(grid, [1, 2]))))
    dist = np.full((h, w), 40.0, dtype=np.float32)
    if positions:
        for sy, sx in positions:
            yy = np.abs(np.arange(h) - sy)[:, None].astype(np.float32)
            xx = np.abs(np.arange(w) - sx)[None, :].astype(np.float32)
            dist = np.minimum(dist, yy + xx)
    channels.append(dist / 40.0)

    # Is coastal — adjacent to ocean (1 ch)
    ocean = (grid == 10)
    coastal = np.zeros((h, w), dtype=np.float32)
    if h > 1:
        coastal[1:] = np.maximum(coastal[1:], ocean[:-1])
        coastal[:-1] = np.maximum(coastal[:-1], ocean[1:])
    if w > 1:
        coastal[:, 1:] = np.maximum(coastal[:, 1:], ocean[:, :-1])
        coastal[:, :-1] = np.maximum(coastal[:, :-1], ocean[:, 1:])
    channels.append(coastal)

    # Is land (1 ch)
    channels.append((grid != 10).astype(np.float32))

    # z broadcast (1 ch)
    channels.append(np.full((h, w), z, dtype=np.float32))

    # Settlement density in 5×5 neighborhood (1 ch) — manual box filter
    settle = np.isin(grid, [1, 2]).astype(np.float32)
    padded = np.pad(settle, 2, mode="constant")
    density = np.zeros((h, w), dtype=np.float32)
    for dy in range(5):
        for dx in range(5):
            density += padded[dy : dy + h, dx : dx + w]
    channels.append(density / 25.0)

    return np.stack(channels)  # (13, H, W)


IN_CHANNELS = 13


def load_dataset(gt_dir, cal_file):
    """Load all GT files into list of (grid, gt, z, round_num)."""
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


def augment(grid, gt, aug_idx):
    """Apply rotation/flip augmentation (0-7)."""
    rot = aug_idx % 4
    flip = aug_idx >= 4
    g = np.rot90(grid, rot).copy()
    t = np.rot90(gt, rot, axes=(0, 1)).copy()
    if flip:
        g = np.fliplr(g).copy()
        t = np.fliplr(t).copy()
    return g, t


def make_batch(samples, augment_on=True):
    """Create a batch from samples with random augmentation."""
    features, targets = [], []
    for grid, gt, z, rn in samples:
        aug = np.random.randint(8) if augment_on else 0
        g, t = augment(grid, gt, aug)
        features.append(encode_grid(g, z))
        targets.append(t)
    return (
        torch.tensor(np.stack(features), dtype=torch.float32).to(DEVICE),
        torch.tensor(np.stack(targets), dtype=torch.float32).to(DEVICE),
    )


# ── Model ─────────────────────────────────────────────────────────────


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class AstarNet(nn.Module):
    """Dilated ResNet — large receptive field without downsampling.

    Receptive field with dilations [1,2,4,8,16,1]: ~67 cells — covers full 40×40.
    """

    def __init__(self, in_ch=IN_CHANNELS, hidden=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(hidden, dilation=1),
            ResBlock(hidden, dilation=2),
            ResBlock(hidden, dilation=4),
            ResBlock(hidden, dilation=8),
            ResBlock(hidden, dilation=16),
            ResBlock(hidden, dilation=1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.stem(x)
        h = self.blocks(h)
        logits = self.head(h)  # (B, 6, H, W)
        return logits.permute(0, 2, 3, 1)  # (B, H, W, 6)


# ── Loss & Scoring ───────────────────────────────────────────────────


def competition_loss(pred_logits, target, eps=1e-8):
    """Entropy-weighted KL divergence — exact competition metric as loss."""
    pred_prob = F.softmax(pred_logits, dim=-1)
    pred_prob = torch.clamp(pred_prob, min=0.01)
    pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)

    # Entropy of target
    entropy = -torch.sum(target * torch.log(torch.clamp(target, min=eps)), dim=-1)

    # KL(target || pred) per cell
    kl = torch.sum(
        target * torch.log(torch.clamp(target, min=eps) / (pred_prob + eps)), dim=-1
    )

    # Entropy-weighted average
    total_entropy = entropy.sum(dim=(-2, -1), keepdim=True) + eps
    weight = entropy / total_entropy
    weighted_kl = (weight * kl).sum(dim=(-2, -1))

    return weighted_kl.mean()


def compute_score(pred_prob_np, gt_np):
    """Compute competition score for one sample."""
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


def train_model(samples, epochs=1000, lr=3e-4, eval_every=100, tag="full"):
    """Train model on given samples, return best model state."""
    model = AstarNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] Model: {n_params:,} params, {len(samples)} samples, {DEVICE}")

    best_score = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        # Shuffle samples into mini-batches
        idx = np.random.permutation(len(samples))
        total_loss = 0
        n_batches = 0
        for start in range(0, len(idx), 8):
            batch_samples = [samples[i] for i in idx[start : start + 8]]
            features, targets = make_batch(batch_samples, augment_on=True)

            pred = model(features)
            loss = competition_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % eval_every == 0:
            model.eval()
            scores = []
            with torch.no_grad():
                features, targets = make_batch(samples, augment_on=False)
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


def loro_cv(samples, epochs=600):
    """Leave-One-Round-Out cross-validation."""
    rounds = sorted(set(s[3] for s in samples))
    print(f"\n{'='*60}")
    print(f"LORO Cross-Validation ({len(rounds)} rounds)")
    print(f"{'='*60}")

    loro_scores = {}
    for test_round in rounds:
        train_s = [s for s in samples if s[3] != test_round]
        test_s = [s for s in samples if s[3] == test_round]

        best_state, train_score = train_model(
            train_s, epochs=epochs, eval_every=200, tag=f"LORO-R{test_round}"
        )

        # Evaluate on held-out round
        model = AstarNet().to(DEVICE)
        model.load_state_dict(best_state)
        model.eval()

        round_scores = []
        with torch.no_grad():
            features, targets = make_batch(test_s, augment_on=False)
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
    print("Phase 1: Full Training (all data)")
    print(f"{'='*60}")
    best_state, best_score = train_model(samples, epochs=1000, eval_every=100, tag="FULL")
    print(f"\nBest full-data score: {best_score:.1f}")

    # Save model
    torch.save(best_state, "astar_nn.pt")
    print(f"Model saved to astar_nn.pt")

    # Phase 2: LORO cross-validation
    loro_scores = loro_cv(samples, epochs=600)

    # Phase 3: Export predictions for comparison
    print(f"\n{'='*60}")
    print("Phase 3: Export predictions")
    print(f"{'='*60}")

    model = AstarNet().to(DEVICE)
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        features, targets = make_batch(samples, augment_on=False)
        pred = model(features)
        pred_prob = F.softmax(pred, dim=-1)
        pred_prob = torch.clamp(pred_prob, min=0.01)
        pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)

        for i, (grid, gt, z, rn) in enumerate(samples):
            s = compute_score(pred_prob[i].cpu().numpy(), targets[i].cpu().numpy())
            seed = i % 5
            print(f"  R{rn} S{seed}: score={s:.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
