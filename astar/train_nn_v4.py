"""Train Conditional U-Net v4 with Global Context Vector.

Key insight: instead of compressing round info into scalar z, extract a
multi-dimensional context vector from observations that captures the round's
hidden parameters (winter severity, food production, raid frequency, etc.)

During training: context vector computed from GT probabilities (exact).
During inference: context vector estimated from 45 observation queries.
Noise injection during training bridges the gap.

Run on A100: python train_nn_v4.py
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

# Context vector dimensions:
# 0: P(settlement survives) for initially-settled cells
# 1: P(port survives) for initially-port cells
# 2: P(ruin) across dynamic cells
# 3: P(forest) across non-ocean cells
# 4: P(empty) for initially-settled cells (collapse indicator)
# 5: P(settlement expands) for initially-empty land cells
# 6: Mean entropy of dynamic cells (round chaos indicator)
# 7: z (settlement alive rate — backward compat)
CONTEXT_DIM = 8


def compute_context_from_gt(grid: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute global context vector from ground truth probabilities."""
    h, w = grid.shape
    ctx = np.zeros(CONTEXT_DIM, dtype=np.float32)

    # Masks
    settlement_mask = (grid == 1)
    port_mask = (grid == 2)
    settled_mask = np.isin(grid, [1, 2])
    ocean_mask = (grid == 10)
    mountain_mask = (grid == 5)
    land_mask = ~ocean_mask & ~mountain_mask
    empty_land_mask = np.isin(grid, [0, 11])  # plains/empty

    # 0: P(settlement survives) — P(class 1) for initially-settled cells
    if settlement_mask.sum() > 0:
        ctx[0] = gt[settlement_mask, 1].mean()
    if port_mask.sum() > 0:
        ctx[0] = (ctx[0] + gt[port_mask, 2].mean()) / 2  # avg of both

    # 1: P(port) for initially-port cells
    if port_mask.sum() > 0:
        ctx[1] = gt[port_mask, 2].mean()

    # 2: P(ruin) across dynamic cells
    if land_mask.sum() > 0:
        ctx[2] = gt[land_mask, 3].mean()

    # 3: P(forest) across non-ocean cells
    if land_mask.sum() > 0:
        ctx[3] = gt[land_mask, 4].mean()

    # 4: P(empty) for initially-settled cells (collapse → empty)
    if settled_mask.sum() > 0:
        ctx[4] = gt[settled_mask, 0].mean()

    # 5: P(settlement+port) for initially-empty land (expansion)
    if empty_land_mask.sum() > 0:
        ctx[5] = (gt[empty_land_mask, 1] + gt[empty_land_mask, 2]).mean()

    # 6: Mean entropy of dynamic cells
    eps = 1e-10
    entropy = -np.sum(gt * np.log(np.clip(gt, eps, 1)), axis=-1)
    dynamic = entropy > 0.01
    if dynamic.sum() > 0:
        ctx[6] = entropy[dynamic].mean() / 1.8  # normalize to ~[0, 1]

    # 7: z (settlement alive rate)
    if settled_mask.sum() > 0:
        ctx[7] = (gt[settled_mask, 1] + gt[settled_mask, 2]).mean()

    return ctx


def add_context_noise(ctx: np.ndarray, rng, noise_level: float = 0.15) -> np.ndarray:
    """Add noise to context vector to simulate observation estimation error."""
    noise = rng.normal(0, noise_level, size=ctx.shape).astype(np.float32)
    noisy = np.clip(ctx + noise, 0, 1)
    return noisy


# ── Grid encoding ────────────────────────────────────────────────────

SPATIAL_CHANNELS = 12  # 8 terrain + dist + coastal + land + density


def encode_grid(grid: np.ndarray) -> np.ndarray:
    """Encode grid → (SPATIAL_CHANNELS, H, W) without z/context."""
    h, w = grid.shape
    channels = []

    for code in TERRAIN_CODES:
        channels.append((grid == code).astype(np.float32))

    # Distance to settlement
    positions = list(zip(*np.where(np.isin(grid, [1, 2]))))
    dist = np.full((h, w), 40.0, dtype=np.float32)
    if positions:
        for sy, sx in positions:
            yy = np.abs(np.arange(h) - sy)[:, None].astype(np.float32)
            xx = np.abs(np.arange(w) - sx)[None, :].astype(np.float32)
            dist = np.minimum(dist, yy + xx)
    channels.append(dist / 40.0)

    # Coastal
    ocean = (grid == 10)
    coastal = np.zeros((h, w), dtype=np.float32)
    if h > 1:
        coastal[1:] = np.maximum(coastal[1:], ocean[:-1])
        coastal[:-1] = np.maximum(coastal[:-1], ocean[1:])
    if w > 1:
        coastal[:, 1:] = np.maximum(coastal[:, 1:], ocean[:, :-1])
        coastal[:, :-1] = np.maximum(coastal[:, :-1], ocean[:, 1:])
    channels.append(coastal)

    # Land
    channels.append((grid != 10).astype(np.float32))

    # Settlement density 5×5
    settle = np.isin(grid, [1, 2]).astype(np.float32)
    padded = np.pad(settle, 2, mode="constant")
    density = np.zeros((h, w), dtype=np.float32)
    for dy in range(5):
        for dx in range(5):
            density += padded[dy : dy + h, dx : dx + w]
    channels.append(density / 25.0)

    return np.stack(channels)


IN_CHANNELS = SPATIAL_CHANNELS + CONTEXT_DIM  # spatial + broadcast context


# ── Dataset ──────────────────────────────────────────────────────────


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

        # Compute context from GT
        ctx = compute_context_from_gt(grid, gt)
        samples.append((grid, gt, ctx, rn))
    return samples


def make_batch(samples, rng, augment=True, noise_ctx=True):
    features, targets = [], []
    for grid, gt, ctx, rn in samples:
        g, t = grid.copy(), gt.copy()

        if augment:
            aug = rng.integers(8)
            rot = aug % 4
            flip = aug >= 4
            g = np.rot90(g, rot).copy()
            t = np.rot90(t, rot, axes=(0, 1)).copy()
            if flip:
                g = np.fliplr(g).copy()
                t = np.fliplr(t).copy()

        # Encode spatial features
        spatial = encode_grid(g)  # (SPATIAL_CHANNELS, H, W)

        # Context: add noise during training to bridge GT→observation gap
        c = add_context_noise(ctx, rng, noise_level=0.12) if noise_ctx else ctx.copy()

        # Broadcast context to spatial dims
        h, w = g.shape
        ctx_broadcast = np.repeat(c[:, None, None], h, axis=1)
        ctx_broadcast = np.repeat(ctx_broadcast, w, axis=2)  # (CONTEXT_DIM, H, W)

        feat = np.concatenate([spatial, ctx_broadcast], axis=0)
        features.append(feat)
        targets.append(t)

    return (
        torch.tensor(np.stack(features), dtype=torch.float32).to(DEVICE),
        torch.tensor(np.stack(targets), dtype=torch.float32).to(DEVICE),
    )


# ── Model ────────────────────────────────────────────────────────────


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


class ConditionalUNet(nn.Module):
    """U-Net conditioned on global context vector.

    Input: (B, SPATIAL_CHANNELS + CONTEXT_DIM, 40, 40)
    Output: (B, 40, 40, 6)
    """

    def __init__(self, in_ch=IN_CHANNELS, hidden=160, num_classes=NUM_CLASSES):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            ResBlock(hidden, dilation=1),
            ResBlock(hidden, dilation=2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(),
            ResBlock(hidden * 2, dilation=1),
            ResBlock(hidden * 2, dilation=2),
            ResBlock(hidden * 2, dilation=4),
        )

        # Bottleneck (10×10 with global context)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(),
            ResBlock(hidden * 2, dilation=1),
            ResBlock(hidden * 2, dilation=2),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(hidden * 2, hidden * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 2, 3, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(),
            ResBlock(hidden * 2, dilation=1),
        )

        self.up1 = nn.ConvTranspose2d(hidden * 2, hidden, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            ResBlock(hidden, dilation=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        e1 = self.enc1(x)       # (B, H, 40, 40)
        e2 = self.enc2(e1)      # (B, 2H, 20, 20)
        b = self.bottleneck(e2)  # (B, 2H, 10, 10)

        d2 = self.up2(b)        # (B, 2H, 20, 20)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)       # (B, H, 40, 40)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1)
        return logits.permute(0, 2, 3, 1)  # (B, 40, 40, 6)


# ── Loss ─────────────────────────────────────────────────────────────


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
        gt_np * np.log(np.clip(gt_np, eps, 1) / np.clip(pred_prob_np, eps, 1)), axis=-1
    )
    w = entropy / (entropy.sum() + eps)
    wkl = (w * kl).sum()
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


# ── Training ─────────────────────────────────────────────────────────


def train_model(samples, epochs=1500, lr=3e-4, tag="full"):
    model = ConditionalUNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    warmup = 50
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (epochs - warmup)
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
            batch = [samples[i] for i in idx[start : start + 8]]
            features, targets = make_batch(batch, rng, augment=True, noise_ctx=True)

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
                features, targets = make_batch(
                    samples, rng, augment=False, noise_ctx=False
                )
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

    # Compute per-round mean context for holdout rounds
    round_contexts = {}
    for rn in rounds:
        ctxs = [s[2] for s in samples if s[3] == rn]
        round_contexts[rn] = np.mean(ctxs, axis=0)

    loro_scores = {}
    for test_round in rounds:
        train_s = [s for s in samples if s[3] != test_round]
        test_s = [s for s in samples if s[3] == test_round]

        best_state, train_score = train_model(
            train_s, epochs=epochs, tag=f"LORO-R{test_round}"
        )

        model = ConditionalUNet().to(DEVICE)
        model.load_state_dict(best_state)
        model.eval()

        rng = np.random.default_rng(0)

        # For holdout: use noisy context to simulate real inference conditions
        # Average context noise over 10 random noise samples
        round_scores_all = []
        for noise_trial in range(5):
            trial_rng = np.random.default_rng(noise_trial + 100)
            noisy_test_s = [
                (g, gt, add_context_noise(ctx, trial_rng, 0.12), rn)
                for g, gt, ctx, rn in test_s
            ]
            with torch.no_grad():
                features, targets = make_batch(
                    noisy_test_s, rng, augment=False, noise_ctx=False
                )
                pred = model(features)
                pred_prob = F.softmax(pred, dim=-1)
                pred_prob = torch.clamp(pred_prob, min=0.01)
                pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)
                for i in range(len(test_s)):
                    s = compute_score(
                        pred_prob[i].cpu().numpy(), targets[i].cpu().numpy()
                    )
                    round_scores_all.append(s)

        avg = np.mean(round_scores_all)
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

    # Show context vectors per round
    rounds = sorted(set(s[3] for s in samples))
    print("\nContext vectors per round:")
    print(f"  {'Round':>5}  {'survive':>8} {'port':>8} {'ruin':>8} {'forest':>8} {'collapse':>8} {'expand':>8} {'entropy':>8} {'z':>8}")
    for rn in rounds:
        ctxs = [s[2] for s in samples if s[3] == rn]
        mean_ctx = np.mean(ctxs, axis=0)
        vals = " ".join(f"{v:8.3f}" for v in mean_ctx)
        print(f"  R{rn:>4}: {vals}")

    # Phase 1: Full training
    print(f"\n{'='*60}")
    print("Phase 1: Full Training v4 (Conditional U-Net)")
    print(f"{'='*60}")
    best_state, best_score = train_model(samples, epochs=1500, tag="FULL")
    print(f"\nBest full-data score: {best_score:.1f}")
    torch.save(best_state, "astar_nn_v4.pt")
    print("Model saved to astar_nn_v4.pt")

    # Phase 2: LORO
    loro_scores = loro_cv(samples, epochs=800)
    print("\nDone!")


if __name__ == "__main__":
    main()
