#!/usr/bin/env python3
"""Config sweep: find optimal blend parameters against all GT.

Pre-computes NN and Dirichlet predictions once, then sweeps blend params
at numpy speed. Runs in seconds, not hours.
"""
import json, numpy as np, logging, itertools, time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)
import nn_predict
from strategy import (
    compute_context_vector, estimate_z_from_context, 
    dirichlet_predict, load_calibration, apply_physics_mask,
    floor_and_normalize, NUM_CLASSES, PROB_FLOOR,
    compute_empirical_observations, empirical_anchor,
)

GT_DIR = Path("ground_truth")
OBS_DIR = Path("observations")

def exact_score(pred, gt):
    eps = 1e-10
    gt = np.asarray(gt, float); pred = np.asarray(pred, float)
    ent = -(gt * np.log(np.clip(gt, eps, 1))).sum(axis=-1)
    kl = (gt * np.log(np.clip(gt, eps, 1) / np.clip(pred, eps, 1))).sum(axis=-1)
    te = ent.sum()
    if te < eps: return 100.0
    return float(100 * np.exp(-3 * ((ent / te) * kl).sum()))

def blend_geometric(nn_pred, dir_pred, nn_weight):
    eps = 1e-8
    dw = 1.0 - nn_weight
    log_blend = nn_weight * np.log(nn_pred + eps) + dw * np.log(dir_pred + eps)
    blended = np.exp(log_blend)
    return floor_and_normalize(blended)

# ── Pre-compute all predictions ──────────────────────────────────────
print("Pre-computing predictions for all rounds...")
t0 = time.time()

CAL = load_calibration()
round_data = {}  # {rn: {seed: {nn_pred, dir_pred, gt, grid, obs_counts, n_obs, z}}}

for rn in range(1, 18):
    obs_dir = OBS_DIR / f"round_{rn}"
    if not obs_dir.exists():
        continue
    bc_file = obs_dir / "base_counts.json"
    bc = json.loads(bc_file.read_text()) if bc_file.exists() else {str(i): 9 for i in range(5)}
    
    initial_grids = []
    all_obs = {}; base_obs = {}
    for s in range(5):
        ig_file = obs_dir / f"initial_seed_{s}.json"
        if not ig_file.exists(): continue
        initial_grids.append(json.loads(ig_file.read_text()))
        o_file = obs_dir / f"observations_seed_{s}.json"
        o = json.loads(o_file.read_text()) if o_file.exists() else []
        all_obs[s] = o
        base_obs[s] = o[:int(bc.get(str(s), len(o)))]
    
    if len(initial_grids) < 5:
        continue
    
    ctx = compute_context_vector(base_obs, initial_grids)
    z = estimate_z_from_context(ctx)
    
    round_data[rn] = {"z": z, "seeds": {}}
    nn_predict._models.clear()  # fresh per round
    
    for s in range(5):
        gt_file = GT_DIR / f"round_{rn}_seed_{s}.json"
        if not gt_file.exists():
            continue
        gt = np.array(json.loads(gt_file.read_text())["ground_truth"], dtype=np.float32)
        grid = initial_grids[s]
        
        # NN prediction (fixed)
        nn_pred = nn_predict.predict(grid, z=z, context=ctx)
        if nn_pred is None:
            continue
        nn_pred = apply_physics_mask(nn_pred, grid)
        
        # Dirichlet prediction (fixed)
        dir_pred = dirichlet_predict(grid, all_obs[s], CAL, z=z)
        dir_pred = apply_physics_mask(dir_pred, grid)
        
        # Observation counts for posterior
        obs_counts, n_observed = compute_empirical_observations(all_obs[s])
        
        round_data[rn]["seeds"][s] = {
            "nn_pred": nn_pred, "dir_pred": dir_pred, "gt": gt,
            "grid": grid, "obs_counts": obs_counts, "n_observed": n_observed,
        }

print(f"Pre-computed {sum(len(rd['seeds']) for rd in round_data.values())} seed predictions in {time.time()-t0:.1f}s")

# ── Sweep parameters ─────────────────────────────────────────────────
NN_HEALTHY_VALUES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
NN_PEAK_VALUES = [0.60, 0.65, 0.70, 0.75]
FLOOR_VALUES = [0.001, 0.003, 0.005]
# Posterior: off, or concentration 30/50/100 with n>=2
POSTERIOR_CONFIGS = [
    ("none", 0, 0),
    ("n2_c30", 2, 30),
    ("n2_c50", 2, 50), 
    ("n2_c100", 2, 100),
    ("n3_c30", 3, 30),
    ("n3_c50", 3, 50),
]

# Regime thresholds
HEALTHY_Z = 0.40
MODERATE_Z = 0.08

print(f"\nSweeping {len(NN_HEALTHY_VALUES)}×{len(NN_PEAK_VALUES)}×{len(FLOOR_VALUES)}×{len(POSTERIOR_CONFIGS)} = "
      f"{len(NN_HEALTHY_VALUES)*len(NN_PEAK_VALUES)*len(FLOOR_VALUES)*len(POSTERIOR_CONFIGS)} configs...")

best_overall = {"score": 0, "config": None}
best_healthy = {"score": 0, "config": None}
best_moderate = {"score": 0, "config": None}
best_weighted = {"score": 0, "config": None}

results = []

for nn_healthy, nn_peak, floor_val, (post_name, min_n, conc) in itertools.product(
    NN_HEALTHY_VALUES, NN_PEAK_VALUES, FLOOR_VALUES, POSTERIOR_CONFIGS
):
    if nn_healthy > nn_peak:
        continue  # invalid: floor > peak
    
    all_scores = []
    healthy_scores = []
    moderate_scores = []
    weighted_scores = []
    
    for rn, rd in round_data.items():
        z = rd["z"]
        regime = "healthy" if z > HEALTHY_Z else "moderate" if z > MODERATE_Z else "catastrophic"
        
        # Compute nn_weight for this z and config
        t1, t2, t3, t4, t5 = 0.05, 0.12, 0.25, 0.35, 0.60
        if z < t1:
            nn_weight = 0.0
        elif z < t2:
            nn_weight = nn_peak * 0.3 * (z - t1) / (t2 - t1)
        elif z < t3:
            nn_weight = nn_peak * (0.3 + 0.7 * (z - t2) / (t3 - t2))
        elif z < t4:
            nn_weight = nn_peak
        elif z < t5:
            frac = (z - t4) / (t5 - t4)
            nn_weight = nn_peak * (1.0 - frac) + nn_healthy * frac
        else:
            nn_weight = nn_healthy
        
        seed_scores = []
        for s, sd in rd["seeds"].items():
            # Blend
            blended = blend_geometric(sd["nn_pred"], sd["dir_pred"], nn_weight)
            
            # Apply floor
            blended = np.maximum(blended, floor_val)
            blended = blended / blended.sum(axis=-1, keepdims=True)
            
            # Optional posterior
            if min_n > 0 and sd["n_observed"].max() >= min_n:
                mask = sd["n_observed"] < min_n
                oc = sd["obs_counts"].copy(); oc[mask] = 0
                no = sd["n_observed"].copy(); no[mask] = 0
                blended = empirical_anchor(blended, oc, no, concentration=conc)
            
            seed_scores.append(exact_score(blended, sd["gt"]))
        
        if seed_scores:
            avg = np.mean(seed_scores)
            all_scores.append(avg)
            weighted_scores.append(avg * 1.05**rn)
            if regime == "healthy":
                healthy_scores.append(avg)
            elif regime == "moderate":
                moderate_scores.append(avg)
    
    if not all_scores:
        continue
    
    overall = np.mean(all_scores)
    healthy = np.mean(healthy_scores) if healthy_scores else 0
    moderate = np.mean(moderate_scores) if moderate_scores else 0
    best_w = max(weighted_scores) if weighted_scores else 0
    
    config = {"nn_healthy": nn_healthy, "nn_peak": nn_peak, "floor": floor_val, 
              "posterior": post_name}
    
    results.append((overall, healthy, moderate, best_w, config))
    
    if overall > best_overall["score"]:
        best_overall = {"score": overall, "config": config, "healthy": healthy, "moderate": moderate}
    if healthy > best_healthy["score"]:
        best_healthy = {"score": healthy, "config": config, "overall": overall}
    if moderate > best_moderate["score"]:
        best_moderate = {"score": moderate, "config": config, "overall": overall}
    if best_w > best_weighted["score"]:
        best_weighted = {"score": best_w, "config": config, "overall": overall}

print(f"\n{'='*70}")
print(f"SWEEP COMPLETE — {len(results)} valid configs tested")
print(f"{'='*70}")
print(f"\nBEST OVERALL AVG: {best_overall['score']:.2f}")
print(f"  Config: {best_overall['config']}")
print(f"  Healthy: {best_overall.get('healthy',0):.2f}, Moderate: {best_overall.get('moderate',0):.2f}")
print(f"\nBEST HEALTHY AVG: {best_healthy['score']:.2f}")
print(f"  Config: {best_healthy['config']}")
print(f"  Overall: {best_healthy.get('overall',0):.2f}")
print(f"\nBEST MODERATE AVG: {best_moderate['score']:.2f}")
print(f"  Config: {best_moderate['config']}")
print(f"  Overall: {best_moderate.get('overall',0):.2f}")
print(f"\nBEST MAX WEIGHTED: {best_weighted['score']:.2f}")
print(f"  Config: {best_weighted['config']}")
print(f"  Overall: {best_weighted.get('overall',0):.2f}")

# Top 10 by overall
results.sort(key=lambda x: -x[0])
print(f"\n{'='*70}")
print(f"TOP 10 CONFIGS BY OVERALL AVG")
print(f"{'='*70}")
print(f"{'Overall':>8} {'Healthy':>8} {'Moderate':>9} {'MaxW':>8} | Config")
for overall, healthy, moderate, best_w, config in results[:10]:
    print(f"{overall:8.2f} {healthy:8.2f} {moderate:9.2f} {best_w:8.1f} | H={config['nn_healthy']:.2f} P={config['nn_peak']:.2f} F={config['floor']} post={config['posterior']}")

# Top 10 by healthy
results.sort(key=lambda x: -x[1])
print(f"\n{'='*70}")
print(f"TOP 10 CONFIGS BY HEALTHY AVG")
print(f"{'='*70}")
print(f"{'Overall':>8} {'Healthy':>8} {'Moderate':>9} {'MaxW':>8} | Config")
for overall, healthy, moderate, best_w, config in results[:10]:
    print(f"{overall:8.2f} {healthy:8.2f} {moderate:9.2f} {best_w:8.1f} | H={config['nn_healthy']:.2f} P={config['nn_peak']:.2f} F={config['floor']} post={config['posterior']}")

# Top 10 by max weighted (what actually matters for leaderboard)
results.sort(key=lambda x: -x[3])
print(f"\n{'='*70}")
print(f"TOP 10 CONFIGS BY MAX WEIGHTED ROUND (LEADERBOARD)")
print(f"{'='*70}")
print(f"{'Overall':>8} {'Healthy':>8} {'Moderate':>9} {'MaxW':>8} | Config")
for overall, healthy, moderate, best_w, config in results[:10]:
    print(f"{overall:8.2f} {healthy:8.2f} {moderate:9.2f} {best_w:8.1f} | H={config['nn_healthy']:.2f} P={config['nn_peak']:.2f} F={config['floor']} post={config['posterior']}")
