#!/usr/bin/env python3
"""Nightforce: H100 overnight training — LORO ensemble + v3-XL.

Phase 1: LORO 16-fold (each fold holds out one round)
Phase 2: v3-XL (256 hidden, 10 blocks) on all data
Phase 3: v3-XL healthy specialist (3x overweight)

All checkpoints saved for morning evaluation.
"""
import json, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from train_nn_v3 import (
    load_dataset, AstarNetV3, train_model, make_batch, compute_score,
    encode_grid, ResBlock, competition_loss, DEVICE, NUM_CLASSES, IN_CHANNELS,
)

GT_DIR = Path('ground_truth')
CAL_FILE = Path('calibration.json')
RESULTS_FILE = Path('nightforce_results.json')


# ── v3-XL: bigger architecture ──────────────────────────────────────

class AstarNetV3XL(nn.Module):
    """V3-XL: 256 hidden, 10 ResBlocks, multi-scale."""
    def __init__(self, in_ch=IN_CHANNELS, hidden=256, num_classes=NUM_CLASSES):
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
            ResBlock(hidden, dilation=16),
            ResBlock(hidden, dilation=8),
            ResBlock(hidden, dilation=4),
            ResBlock(hidden, dilation=2),
            ResBlock(hidden, dilation=1),
        )
        self.down_conv = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
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
        return self.head(merged).permute(0, 2, 3, 1)


def train_model_xl(samples, epochs=800, lr=3e-4, tag='XL'):
    """Train v3-XL model (same as train_model but with XL arch)."""
    model = AstarNetV3XL().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    warmup_epochs = 50
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{tag}] Model: {n_params:,} params, {len(samples)} samples, {DEVICE}')

    rng = np.random.default_rng(42)
    best_score = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(samples))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(idx), 8):
            batch_samples = [samples[i] for i in idx[start:start + 8]]
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

        if (epoch + 1) % 50 == 0:
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
            mn = np.min(scores)
            print(f'  [{tag}] Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, avg={avg:.1f}, min={mn:.1f}, lr={scheduler.get_last_lr()[0]:.6f}')
            if avg > best_score:
                best_score = avg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_score


def run_loro(samples):
    """Phase 1: LORO 16-fold cross-validation with checkpoint saving."""
    cal = json.loads(CAL_FILE.read_text())
    round_z = {int(k): v for k, v in cal.get('round_z', {}).items()}
    rounds = sorted(set(s[3] for s in samples))

    print(f'\n{"="*60}')
    print(f'PHASE 1: LORO Cross-Validation ({len(rounds)} folds)')
    print(f'{"="*60}')

    loro_scores = {}
    for test_round in rounds:
        train_s = [s for s in samples if s[3] != test_round]
        test_s = [s for s in samples if s[3] == test_round]
        z = round_z.get(test_round, 0.283)
        regime = 'healthy' if z > 0.40 else 'moderate' if z > 0.15 else 'catastrophic'

        best_state, train_score = train_model(
            train_s, epochs=400, tag=f'LORO-R{test_round}'
        )

        # Save fold checkpoint
        torch.save(best_state, f'loro_fold_{test_round}.pt')

        # Evaluate on held-out round
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
                s = compute_score(pred_prob[i].cpu().numpy(), targets[i].cpu().numpy())
                round_scores.append(s)

        avg = np.mean(round_scores)
        loro_scores[test_round] = {
            'avg': float(avg),
            'seeds': [float(s) for s in round_scores],
            'train_fit': float(train_score),
            'z': z,
            'regime': regime,
        }
        print(f'  R{test_round:2d} ({regime:12s} z={z:.3f}): LORO={avg:.1f}  train_fit={train_score:.1f}  gap={train_score-avg:.1f}')

    overall = np.mean([v['avg'] for v in loro_scores.values()])
    print(f'\nLORO average: {overall:.1f}')
    return loro_scores


def run_xl(samples):
    """Phase 2: v3-XL on all data."""
    print(f'\n{"="*60}')
    print('PHASE 2: v3-XL (256 hidden, 10 blocks)')
    print(f'{"="*60}')

    best_state, best_score = train_model_xl(samples, epochs=800, tag='v3XL')
    torch.save(best_state, 'v3_xl.pt')
    print(f'v3-XL saved. Best training score: {best_score:.1f}')
    return best_score


def run_xl_healthy(samples):
    """Phase 2b: v3-XL healthy specialist."""
    cal = json.loads(CAL_FILE.read_text())
    round_z = {int(k): v for k, v in cal.get('round_z', {}).items()}
    healthy_rounds = {rn for rn, z in round_z.items() if z > 0.40}

    print(f'\n{"="*60}')
    print(f'PHASE 2b: v3-XL Healthy Specialist (3x weight on {sorted(healthy_rounds)})')
    print(f'{"="*60}')

    weighted = []
    for s in samples:
        if s[3] in healthy_rounds:
            weighted.extend([s] * 3)
        else:
            weighted.append(s)
    print(f'Weighted: {len(weighted)} samples ({len(samples)} base)')

    best_state, best_score = train_model_xl(weighted, epochs=800, tag='v3XL-H')
    torch.save(best_state, 'v3_xl_healthy.pt')
    print(f'v3-XL-Healthy saved. Best training score: {best_score:.1f}')
    return best_score


def main():
    t0 = time.time()
    samples = load_dataset(GT_DIR, CAL_FILE)
    print(f'Loaded {len(samples)} GT samples')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    results = {}

    if mode in ('all', 'loro'):
        loro = run_loro(samples)
        results['loro'] = loro

    if mode in ('all', 'xl'):
        xl_score = run_xl(samples)
        results['xl_score'] = xl_score
        xl_h_score = run_xl_healthy(samples)
        results['xl_healthy_score'] = xl_h_score

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*60}')
    print(f'NIGHTFORCE COMPLETE — {elapsed:.0f} min total')
    print(f'Results saved to {RESULTS_FILE}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
