"""Retrain v2 (AstarNet, 128 hidden, 6 dilated ResBlocks) on 90 GT samples.
Uses train_nn_v3.py recipe: competition_loss, z-augmentation, warmup+cosine, 1500 epochs.
Encode_grid matches nn_predict.py encode_grid_v2v3 exactly.
"""
import json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('v2_retrain')

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

GT_DIR = Path("ground_truth")
CAL_FILE = Path("calibration.json")
NUM_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
IN_CHANNELS = 13


def encode_grid(grid, z):
    """MUST match encode_grid_v2v3 in nn_predict.py exactly."""
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
            density += padded[dy:dy+h, dx:dx+w]
    channels.append(density / 25.0)
    return np.stack(channels)


# ── v2 Architecture (exact match to nn_predict.py AstarNet) ──────────

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
    """V2: 128 hidden, 6 dilated ResBlocks, ReLU, dropout in head."""
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
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h).permute(0, 2, 3, 1)


# ── Loss + Scoring ───────────────────────────────────────────────────

def competition_loss(pred_logits, target, eps=1e-8):
    pred_prob = F.softmax(pred_logits, dim=-1)
    pred_prob = torch.clamp(pred_prob, min=0.01)
    pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(target * torch.log(torch.clamp(target, min=eps)), dim=-1)
    kl = torch.sum(target * torch.log(torch.clamp(target, min=eps) / (pred_prob + eps)), dim=-1)
    total_entropy = entropy.sum(dim=(-2, -1), keepdim=True) + eps
    weight = entropy / total_entropy
    return (weight * kl).sum(dim=(-2, -1)).mean()


def compute_score(pred, gt):
    eps = 1e-10
    gt = np.clip(gt, eps, 1.0)
    pred = np.clip(pred, eps, 1.0)
    entropy = -np.sum(gt * np.log(gt), axis=-1)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    w = entropy / (entropy.sum() + eps)
    return max(0, min(100, 100 * np.exp(-3 * (w * kl).sum())))


# ── Data ─────────────────────────────────────────────────────────────

def load_dataset():
    cal = json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else {}
    round_z = {int(k): v for k, v in cal.get("round_z", {}).items()}
    samples = []
    for gt_file in sorted(GT_DIR.glob("round_*_seed_*.json")):
        data = json.loads(gt_file.read_text())
        rn = int(gt_file.stem.split("_")[1])
        grid = np.array(data["initial_grid"], dtype=np.int32)
        gt = np.array(data["ground_truth"], dtype=np.float32)
        z = round_z.get(rn, 0.283)
        samples.append((grid, gt, z, rn))
    return samples, round_z


def z_augment_sample(grid, gt, z, target_z, rng):
    if abs(target_z - z) < 0.05:
        return gt.copy()
    shift = target_z - z
    new_gt = gt.copy()
    h, w, c = gt.shape
    for y in range(h):
        for x in range(w):
            if grid[y, x] in (10, 5):
                continue
            p = new_gt[y, x].copy()
            if shift < 0:
                transfer = min(p[1], -shift * 0.5)
                p[1] -= transfer; p[3] += transfer * 0.3; p[0] += transfer * 0.7
                transfer_p = min(p[2], -shift * 0.5)
                p[2] -= transfer_p; p[3] += transfer_p * 0.3; p[0] += transfer_p * 0.7
                transfer_r = min(p[3], -shift * 0.3)
                p[3] -= transfer_r; p[4] += transfer_r * 0.5; p[0] += transfer_r * 0.5
            else:
                transfer = min(p[0] * 0.3, shift * 0.3)
                p[0] -= transfer
                if grid[y, x] in (1, 2, 11):
                    p[1] += transfer * 0.7; p[2] += transfer * 0.3
                else:
                    p[4] += transfer
                transfer_r = min(p[3], shift * 0.3)
                p[3] -= transfer_r; p[1] += transfer_r * 0.5; p[0] += transfer_r * 0.5
            p = np.clip(p, 0.001, None)
            new_gt[y, x] = p / p.sum()
    return new_gt


def make_batch(samples, rng, augment=True, z_augment=True):
    features, targets = [], []
    for grid, gt, z, rn in samples:
        g, t = grid.copy(), gt.copy()
        if augment:
            aug = rng.integers(8)
            rot = aug % 4; flip = aug >= 4
            g = np.rot90(g, rot).copy(); t = np.rot90(t, rot, axes=(0, 1)).copy()
            if flip:
                g = np.fliplr(g).copy(); t = np.fliplr(t).copy()
        z_used = z
        if z_augment and augment and rng.random() < 0.3:
            target_z = rng.uniform(0.01, 0.45)
            t = z_augment_sample(g, t, z, target_z, rng)
            z_used = target_z
        features.append(encode_grid(g, z_used))
        targets.append(t)
    return (torch.tensor(np.stack(features), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.stack(targets), dtype=torch.float32).to(DEVICE))


# ── Training ─────────────────────────────────────────────────────────

def train_model(samples, epochs=1500, lr=3e-4, tag="full"):
    model = AstarNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    warmup_epochs = 50

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[{tag}] {n_params:,} params, {len(samples)} samples, {DEVICE}")

    rng = np.random.default_rng(42)
    best_score = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(samples))
        total_loss = 0; n_batches = 0

        for start in range(0, len(idx), 8):
            batch_samples = [samples[i] for i in idx[start:start + 8]]
            features, targets = make_batch(batch_samples, rng, augment=True, z_augment=True)
            pred = model(features)
            loss = competition_loss(pred, targets)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

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
                    s = compute_score(pred_prob[i].cpu().numpy(), targets[i].cpu().numpy())
                    scores.append(s)
            avg = np.mean(scores)
            log.info(f"  [{tag}] Ep {epoch+1}: loss={total_loss/n_batches:.4f}, avg={avg:.1f}, lr={scheduler.get_last_lr()[0]:.6f}")
            if avg > best_score:
                best_score = avg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log.info(f"    BEST: {best_score:.1f}")

    return best_state, best_score


def main():
    log.info("=" * 60)
    log.info("RETRAIN V2 (AstarNet) on R1-R18 (90 samples)")
    log.info("=" * 60)

    samples, round_z = load_dataset()
    log.info(f"Loaded {len(samples)} samples")

    # Phase 1: Full training
    log.info("\n=== PHASE 1: Full Training ===")
    best_state, best_score = train_model(samples, epochs=1500, tag="FULL")
    log.info(f"Best full-data score: {best_score:.1f}")
    torch.save(best_state, "astar_nn_v2_r18.pt")
    log.info("Saved astar_nn_v2_r18.pt")

    # Phase 2: Quick LORO on late rounds only (R11+) to validate
    log.info("\n=== PHASE 2: LORO (late rounds R11+) ===")
    late_rounds = sorted(set(s[3] for s in samples if s[3] >= 11))
    loro = {}
    for holdout in late_rounds:
        train_s = [s for s in samples if s[3] != holdout]
        test_s = [s for s in samples if s[3] == holdout]
        fold_state, _ = train_model(train_s, epochs=800, tag=f"LORO-R{holdout}")

        model = AstarNet().to(DEVICE)
        model.load_state_dict(fold_state)
        model.eval()
        rng = np.random.default_rng(0)
        with torch.no_grad():
            features, targets = make_batch(test_s, rng, augment=False, z_augment=False)
            pred = model(features)
            pred_prob = F.softmax(pred, dim=-1)
            pred_prob = torch.clamp(pred_prob, min=0.01)
            pred_prob = pred_prob / pred_prob.sum(dim=-1, keepdim=True)
            fold_scores = []
            for i in range(len(test_s)):
                fold_scores.append(compute_score(pred_prob[i].cpu().numpy(), targets[i].cpu().numpy()))

        z = round_z.get(holdout, 0.283)
        regime = 'healthy' if z > 0.40 else 'moderate' if z > 0.15 else 'catastrophic'
        avg = np.mean(fold_scores)
        loro[holdout] = avg
        log.info(f"  R{holdout} z={z:.3f} {regime}: LORO={avg:.1f}")

    log.info(f"\nLate LORO avg: {np.mean(list(loro.values())):.1f}")
    log.info("Done!")


if __name__ == "__main__":
    main()
