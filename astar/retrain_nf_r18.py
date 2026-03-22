"""Retrain NF healthy specialist on expanded R1-R18 dataset (90 samples).
Uses EXACT recipe from nightforce_v2.py: AstarNetV3 with GELU, z_jitter=0.06,
healthy weighting (3x for z>0.40), lr=2e-3, CosineAnnealingLR, 600 epochs.
"""
import json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('retrain')

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


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class AstarNetV3(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, hidden=192, n_blocks=8, n_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.down = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            ResBlock(hidden),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden, n_classes, 1)

    def forward(self, x):
        h = self.stem(x)
        h1 = self.blocks(h)
        h2 = self.down(h1)
        if h2.shape != h1.shape:
            h2 = F.interpolate(h2, size=h1.shape[2:], mode='bilinear', align_corners=False)
        merged = self.merge(torch.cat([h1, h2], dim=1))
        return self.head(merged)


def competition_loss(logits, targets):
    pred = F.softmax(logits, dim=-1)
    pred = torch.clamp(pred, min=1e-6)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    eps = 1e-10
    gt = torch.clamp(targets, min=eps)
    kl = (gt * torch.log(gt / pred)).sum(dim=-1)
    entropy = -(gt * torch.log(gt)).sum(dim=-1)
    total_ent = entropy.sum() + eps
    weights = entropy / total_ent
    return (weights * kl).sum()


def compute_score(pred, gt):
    eps = 1e-10
    gt = np.clip(gt, eps, 1.0)
    pred = np.clip(pred, eps, 1.0)
    entropy = -np.sum(gt * np.log(gt), axis=-1)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    total_ent = entropy.sum()
    if total_ent < eps:
        return 100.0
    w = entropy / (total_ent + eps)
    return max(0, min(100, 100 * np.exp(-3 * (w * kl).sum())))


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
    return samples


def make_batch(samples, rng, augment=True, z_jitter=0.06):
    features, targets = [], []
    for grid, gt, z, rn in samples:
        g, t = grid.copy(), gt.copy()
        if augment:
            aug = rng.integers(8)
            rot = aug % 4
            flip = aug >= 4
            if rot:
                g = np.rot90(g, rot).copy()
                t = np.rot90(t, rot).copy()
            if flip:
                g = np.fliplr(g).copy()
                t = np.fliplr(t).copy()
        z_aug = z
        if augment and z_jitter > 0:
            z_aug = z + rng.normal(0, z_jitter)
            z_aug = np.clip(z_aug, 0.0, 1.0)
        features.append(encode_grid(g, z_aug))
        targets.append(t)
    return (torch.tensor(np.stack(features), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(targets), dtype=torch.float32, device=DEVICE))


def train_family(name, train_samples, eval_samples, sample_weights=None,
                 epochs=600, z_jitter=0.06, lr=2e-3, batch_size=16):
    model = AstarNetV3().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[{name}] {n_params:,} params, {len(train_samples)} train, {len(eval_samples)} eval, {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    rng = np.random.default_rng(42)

    best_eval = 0
    best_state = None
    patience = 0
    max_patience = 80

    for epoch in range(1, epochs + 1):
        model.train()
        if sample_weights is not None:
            probs = np.array(sample_weights, dtype=np.float64)
            probs /= probs.sum()
            indices = rng.choice(len(train_samples), size=min(batch_size, len(train_samples)),
                                 replace=True, p=probs)
            batch = [train_samples[i] for i in indices]
        else:
            rng.shuffle(train_samples)
            batch = train_samples[:batch_size]

        feat, tgt = make_batch(batch, rng, augment=True, z_jitter=z_jitter)
        logits = model(feat).permute(0, 2, 3, 1)
        loss = competition_loss(logits, tgt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                eval_scores = []
                for grid, gt, z, rn in eval_samples:
                    feat_e = torch.tensor(encode_grid(grid, z)[None], dtype=torch.float32, device=DEVICE)
                    logits_e = model(feat_e).permute(0, 2, 3, 1)
                    pred = F.softmax(logits_e, dim=-1)[0].cpu().numpy()
                    pred = np.clip(pred, 0.003, None)
                    pred = pred / pred.sum(axis=-1, keepdims=True)
                    eval_scores.append((rn, compute_score(pred, gt)))

                round_scores = {}
                for rn, s in eval_scores:
                    round_scores.setdefault(rn, []).append(s)
                avg_by_round = {rn: np.mean(ss) for rn, ss in round_scores.items()}
                overall = np.mean(list(avg_by_round.values()))

                if overall > best_eval:
                    best_eval = overall
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                    marker = " ***"
                else:
                    patience += 1
                    marker = ""

                if epoch % 100 == 0 or epoch == 1 or marker:
                    log.info(f"  E{epoch:4d} loss={loss.item():.4f} avg={overall:.2f}{marker}")

                if patience >= max_patience:
                    log.info(f"  Early stop at epoch {epoch} (patience={max_patience})")
                    break

    log.info(f"  BEST: {name} avg={best_eval:.2f}")
    return best_state, best_eval


def forward_validate(model_state, samples):
    model = AstarNetV3().to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()
    scores = []
    with torch.no_grad():
        for grid, gt, z, rn in samples:
            feat = torch.tensor(encode_grid(grid, z)[None], dtype=torch.float32, device=DEVICE)
            logits = model(feat).permute(0, 2, 3, 1)
            pred = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = np.clip(pred, 0.003, None)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            scores.append(compute_score(pred, gt))
    return np.mean(scores) if scores else 0


def main():
    log.info("=" * 60)
    log.info("RETRAIN NF HEALTHY on R1-R18 (90 samples)")
    log.info("=" * 60)

    all_samples = load_dataset()
    log.info(f"Loaded {len(all_samples)} GT samples")

    cal = json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else {}
    round_z = {int(k): v for k, v in cal.get("round_z", {}).items()}
    rounds = sorted(set(s[3] for s in all_samples))
    for rn in rounds:
        z = round_z.get(rn, 0.283)
        regime = 'healthy' if z > 0.40 else 'moderate' if z > 0.15 else 'catastrophic'
        count = sum(1 for s in all_samples if s[3] == rn)
        log.info(f"  R{rn:2d}: z={z:.3f} {regime} ({count} seeds)")

    # ── Phase 1: Train healthy specialist on ALL 90 samples ───────
    log.info("\n=== PHASE 1: Healthy Specialist (all data) ===")
    healthy_weights = []
    for grid, gt, z, rn in all_samples:
        healthy_weights.append(3.0 if z > 0.40 else 1.0)

    state, score = train_family(
        "Healthy-All-R18", all_samples, all_samples,
        sample_weights=healthy_weights,
        epochs=600, z_jitter=0.06,
    )
    torch.save(state, "nf2_healthy_r18.pt")
    log.info(f"Saved nf2_healthy_r18.pt (in-sample avg={score:.2f})")

    # ── Phase 2: LORO to validate improvement ─────────────────────
    log.info("\n=== PHASE 2: LORO Cross-Validation ===")
    loro = {}
    for holdout_round in rounds:
        train_s = [s for s in all_samples if s[3] != holdout_round]
        test_s = [s for s in all_samples if s[3] == holdout_round]
        if not test_s:
            continue

        hw = []
        for g, gt, z, rn in train_s:
            hw.append(3.0 if z > 0.40 else 1.0)

        fold_state, fold_train = train_family(
            f"LORO-R{holdout_round}", train_s, train_s,
            sample_weights=hw, epochs=600, z_jitter=0.06,
        )
        fold_score = forward_validate(fold_state, test_s)
        z = round_z.get(holdout_round, 0.283)
        regime = 'healthy' if z > 0.40 else 'moderate' if z > 0.15 else 'catastrophic'
        loro[holdout_round] = fold_score
        log.info(f"  R{holdout_round} (z={z:.3f} {regime}): LORO={fold_score:.2f}")

    # Summary
    log.info("\n=== LORO SUMMARY ===")
    healthy_scores = [v for k, v in loro.items() if round_z.get(k, 0.283) > 0.40]
    moderate_scores = [v for k, v in loro.items() if 0.15 < round_z.get(k, 0.283) <= 0.40]
    catastrophic_scores = [v for k, v in loro.items() if round_z.get(k, 0.283) <= 0.15]

    for rn in sorted(loro):
        z = round_z.get(rn, 0.283)
        regime = 'healthy' if z > 0.40 else 'moderate' if z > 0.15 else 'catastrophic'
        log.info(f"  R{rn:2d} z={z:.3f} {regime:>12s}: {loro[rn]:.2f}")

    log.info(f"\n  Overall avg: {np.mean(list(loro.values())):.2f}")
    if healthy_scores:
        log.info(f"  Healthy avg: {np.mean(healthy_scores):.2f}")
    if moderate_scores:
        log.info(f"  Moderate avg: {np.mean(moderate_scores):.2f}")
    if catastrophic_scores:
        log.info(f"  Catastrophic avg: {np.mean(catastrophic_scores):.2f}")
    log.info("\nDone!")


if __name__ == "__main__":
    main()
