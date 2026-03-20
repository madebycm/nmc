"""Phase 4: Train AstarNet on replay data with auxiliary heads.

MC-averaged replay frame 50 as targets (official GT preferred when available).
Auxiliary scalar heads: z, port_rate, ruin_fraction. Round-balanced + LORO.
Run on A100: source /astar/venv/bin/activate && python3 train_nn_replay.py
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

REPLAY_DIR, GT_DIR, CAL_FILE = Path("replays"), Path("ground_truth"), Path("calibration.json")
NUM_CLASSES, IN_CHANNELS = 6, 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
CODE_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# ── Encoding (matches nn_predict.py encode_grid_v2v3 exactly) ────────

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
            density += padded[dy:dy+h, dx:dx+w]
    channels.append(density / 25.0)
    return np.stack(channels)

# ── Data loading ─────────────────────────────────────────────────────

def grid_to_probs(grid):
    h, w = grid.shape
    probs = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for code, cls in CODE_TO_CLASS.items():
        probs[:, :, cls] += (grid == code).astype(np.float32)
    return probs

def compute_sample_stats(frames):
    g0, g50 = np.array(frames[0]["grid"]), np.array(frames[50]["grid"])
    s0, s50 = np.isin(g0, [1, 2]), np.isin(g50, [1, 2])
    n_init = s0.sum()
    z = float((s0 & s50).sum() / max(n_init, 1))
    p0 = (g0 == 2)
    np0 = p0.sum()
    port_rate = float((p0 & (g50 == 2)).sum() / max(np0, 1)) if np0 > 0 else 0.5
    land = (g50 != 10).sum()
    ruin_frac = float((g50 == 3).sum() / max(land, 1))
    return z, port_rate, ruin_frac

def load_replay_dataset():
    round_z = {}
    if CAL_FILE.exists():
        cal = json.loads(CAL_FILE.read_text())
        round_z = {int(k): v for k, v in cal.get("round_z", {}).items()}
    gt_data = {}
    for gf in sorted(GT_DIR.glob("round_*_seed_*.json")):
        parts = gf.stem.split("_")
        rn, sn = int(parts[1]), int(parts[3])
        data = json.loads(gf.read_text())
        gt_data[(rn, sn)] = {
            "grid": np.array(data["initial_grid"], dtype=np.int32),
            "target": np.array(data["ground_truth"], dtype=np.float32),
        }
    configs = {}
    for rd in sorted(REPLAY_DIR.glob("round_*")):
        rn = int(rd.name.split("_")[1])
        for sd in sorted(rd.glob("seed_*")):
            sn = int(sd.name.split("_")[1])
            sfiles = sorted(sd.glob("sample_*.json"))
            if not sfiles:
                continue
            mc_probs, zs, prs, rfs = [], [], [], []
            for sf in sfiles:
                d = json.loads(sf.read_text())
                mc_probs.append(grid_to_probs(np.array(d["frames"][50]["grid"])))
                zv, pv, rv = compute_sample_stats(d["frames"])
                zs.append(zv); prs.append(pv); rfs.append(rv)
            first = json.loads(sfiles[0].read_text())
            init_grid = np.array(first["frames"][0]["grid"], dtype=np.int32)
            mc_target = np.mean(mc_probs, axis=0)
            if (rn, sn) in gt_data:
                target, init_grid = gt_data[(rn, sn)]["target"], gt_data[(rn, sn)]["grid"]
            else:
                target = mc_target
            target = np.clip(target, 0.001, None)
            target /= target.sum(axis=-1, keepdims=True)
            configs[(rn, sn)] = dict(
                grid=init_grid, target=target, z=float(np.mean(zs)),
                port_rate=float(np.mean(prs)), ruin_frac=float(np.mean(rfs)),
                round_num=rn, seed=sn, n_samples=len(sfiles))
    for (rn, sn), gtd in gt_data.items():
        if (rn, sn) not in configs:
            configs[(rn, sn)] = dict(
                grid=gtd["grid"], target=gtd["target"], z=round_z.get(rn, 0.283),
                port_rate=0.5, ruin_frac=0.02, round_num=rn, seed=sn, n_samples=0)
    samples = list(configs.values())
    print(f"Loaded {len(samples)} configs from {len(set(s['round_num'] for s in samples))} rounds")
    for rn in sorted(set(s["round_num"] for s in samples)):
        rs = [s for s in samples if s["round_num"] == rn]
        ns = sum(s["n_samples"] for s in rs)
        print(f"  R{rn}: {len(rs)} seeds, {ns} replays, z={np.mean([s['z'] for s in rs]):.3f}")
    return samples

# ── Model ────────────────────────────────────────────────────────────

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

class AstarNetAux(nn.Module):
    """AstarNet v2 backbone + 3 auxiliary scalar heads."""
    def __init__(self, in_ch=IN_CHANNELS, hidden=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU())
        self.blocks = nn.Sequential(
            ResBlock(hidden, 1), ResBlock(hidden, 2), ResBlock(hidden, 4),
            ResBlock(hidden, 8), ResBlock(hidden, 16), ResBlock(hidden, 1))
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 1), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1))
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_z = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.aux_port = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.aux_ruin = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        logits = self.head(h).permute(0, 2, 3, 1)
        pooled = self.aux_pool(h).flatten(1)
        return (logits,
                torch.sigmoid(self.aux_z(pooled)).squeeze(-1),
                torch.sigmoid(self.aux_port(pooled)).squeeze(-1),
                torch.sigmoid(self.aux_ruin(pooled)).squeeze(-1))

    def export_base(self):
        """State dict compatible with base AstarNet (strips aux heads)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("aux_")}

# ── Loss & Scoring ───────────────────────────────────────────────────

def competition_loss(pred_logits, target, eps=1e-8):
    pred = F.softmax(pred_logits, dim=-1)
    pred = torch.clamp(pred, min=0.01)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    ent = -torch.sum(target * torch.log(torch.clamp(target, min=eps)), dim=-1)
    kl = torch.sum(target * torch.log(torch.clamp(target, min=eps) / (pred + eps)), dim=-1)
    w = ent / (ent.sum(dim=(-2, -1), keepdim=True) + eps)
    return (w * kl).sum(dim=(-2, -1)).mean()

def compute_score(pred_np, gt_np):
    eps = 1e-10
    ent = -np.sum(gt_np * np.log(np.clip(gt_np, eps, 1)), axis=-1)
    kl = np.sum(gt_np * np.log(np.clip(gt_np, eps, 1) / np.clip(pred_np, eps, 1)), axis=-1)
    w = ent / (ent.sum() + eps)
    return max(0, min(100, 100 * np.exp(-3 * (w * kl).sum())))

# ── Augmentation & batching ──────────────────────────────────────────

def augment(grid, target, rng):
    rot, flip = rng.integers(4), rng.random() < 0.5
    g = np.rot90(grid, rot).copy()
    t = np.rot90(target, rot, axes=(0, 1)).copy()
    if flip:
        g, t = np.fliplr(g).copy(), np.fliplr(t).copy()
    return g, t

def make_batch(samples, rng, aug=True, z_noise=0.05):
    feats, tgts, zv, pv, rv = [], [], [], [], []
    for s in samples:
        grid, target, z = s["grid"], s["target"], s["z"]
        if aug:
            grid, target = augment(grid, target, rng)
            z = float(np.clip(z + rng.uniform(-z_noise, z_noise), 0.0, 1.0))
        feats.append(encode_grid(grid, z))
        tgts.append(target)
        zv.append(s["z"]); pv.append(s["port_rate"]); rv.append(s["ruin_frac"])
    T = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=DEVICE)
    return T(np.stack(feats)), T(np.stack(tgts)), T(zv), T(pv), T(rv)

def round_balanced_indices(samples, rng, batch_size=32):
    rounds = sorted(set(s["round_num"] for s in samples))
    by_round = {r: [i for i, s in enumerate(samples) if s["round_num"] == r] for r in rounds}
    per_round = max(1, batch_size // len(rounds))
    n_batches = max(1, max(len(v) for v in by_round.values()) // per_round)
    for _ in range(n_batches):
        batch = []
        for r in rounds:
            idx = by_round[r]
            batch.extend(rng.choice(idx, min(per_round, len(idx)), replace=len(idx) < per_round).tolist())
        rng.shuffle(batch)
        yield batch[:batch_size]

# ── Training ─────────────────────────────────────────────────────────

def evaluate(model, samples, rng, tag, epoch, loss):
    model.eval()
    with torch.no_grad():
        feat, tgt, _, _, _ = make_batch(samples, rng, aug=False, z_noise=0)
        logits = model(feat)[0]
        pp = F.softmax(logits, dim=-1)
        pp = torch.clamp(pp, min=0.01)
        pp = pp / pp.sum(dim=-1, keepdim=True)
        scores = [compute_score(pp[i].cpu().numpy(), tgt[i].cpu().numpy()) for i in range(len(samples))]
    avg = np.mean(scores)
    print(f"  [{tag}] Ep {epoch}: loss={loss:.4f}, avg={avg:.1f}, min={np.min(scores):.1f}")
    rounds = sorted(set(s["round_num"] for s in samples))
    parts = [f"R{r}={np.mean([scores[i] for i,s in enumerate(samples) if s['round_num']==r]):.1f}" for r in rounds]
    print(f"         {' '.join(parts)}")
    return avg

def train(samples, epochs=300, lr=1e-3, tag="full", eval_samples=None):
    model = AstarNetAux().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"[{tag}] {sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(samples)} samples, {DEVICE}")
    rng = np.random.default_rng(42)
    best_score, best_state = 0, None

    for epoch in range(epochs):
        model.train()
        tloss, nb = 0, 0
        for bidx in round_balanced_indices(samples, rng, 32):
            feat, tgt, zt, pt, rt = make_batch([samples[i] for i in bidx], rng, aug=True)
            logits, zp, pp, rp = model(feat)
            loss = competition_loss(logits, tgt) + 0.1 * (
                F.mse_loss(zp, zt) + F.mse_loss(pp, pt) + F.mse_loss(rp, rt))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item(); nb += 1
        sched.step()
        if (epoch + 1) % 50 == 0:
            ev = eval_samples if eval_samples is not None else samples
            score = evaluate(model, ev, rng, tag, epoch+1, tloss/max(nb, 1))
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_score

# ── LORO ─────────────────────────────────────────────────────────────

def loro_cv(all_samples, epochs=300):
    rounds = sorted(set(s["round_num"] for s in all_samples))
    print(f"\n{'='*60}\nLORO Cross-Validation ({len(rounds)} rounds)\n{'='*60}")
    loro = {}
    for rn in rounds:
        tr = [s for s in all_samples if s["round_num"] != rn]
        te = [s for s in all_samples if s["round_num"] == rn]
        if not tr or not te:
            continue
        _, best = train(tr, epochs=epochs, tag=f"LORO-R{rn}", eval_samples=te)
        loro[rn] = best
        print(f"  => R{rn} holdout: {best:.1f}")
    if loro:
        print(f"\nLORO avg: {np.mean(list(loro.values())):.1f}")
        for rn in sorted(loro):
            print(f"  R{rn}: {loro[rn]:.1f}")
    return loro

# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    samples = load_replay_dataset()
    if not samples:
        print("No data!"); return

    print(f"\n{'='*60}\nPhase 1: Full Training (replay + GT)\n{'='*60}")
    best_state, best_score = train(samples, epochs=300, tag="FULL")
    print(f"\nBest score: {best_score:.1f}")

    torch.save(best_state, "astar_nn_replay_full.pt")
    print("Saved astar_nn_replay_full.pt (with aux heads)")

    model = AstarNetAux()
    model.load_state_dict(best_state)
    torch.save(model.export_base(), "astar_nn_replay.pt")
    print("Saved astar_nn_replay.pt (base AstarNet compatible)")

    loro_cv(samples, epochs=300)
    print("\nDone!")

if __name__ == "__main__":
    main()
