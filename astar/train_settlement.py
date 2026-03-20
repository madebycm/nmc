"""Phase 3: Settlement survival model from replay data.

Predicts settlement alive/dead at step 50 from step 0 features.
Uses XGBoost (fast, tabular). LORO validation (hold out entire rounds).

Features per settlement:
    population, food, wealth, defense, has_port,
    x, y, n_neighbors, near_coast, near_mountain, near_forest,
    faction_size (# settlements owned by same owner)

Also outputs round-level z estimate = mean(predicted_survival).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ROOT = Path(__file__).parent
REPLAY_DIR = ROOT / "replays"
TERRAIN_CODES = {"ocean": [0, 10], "land": [11], "settlement": [1], "port": [2],
                 "ruin": [3], "forest": [4], "mountain": [5]}


def extract_settlement_features(frame0, grid0, all_settlements_0):
    """Extract features for each settlement from step 0."""
    h, w = len(grid0), len(grid0[0])
    grid = np.array(grid0)

    features = []
    for s in frame0["settlements"]:
        if not s["alive"]:
            continue
        x, y = s["x"], s["y"]
        pop = s["population"]
        food = s["food"]
        wealth = s["wealth"]
        defense = s["defense"]
        has_port = 1.0 if s["has_port"] else 0.0

        # Local terrain context (3x3 neighborhood)
        near_coast = 0
        near_mountain = 0
        near_forest = 0
        n_neighbors = 0  # settlement/port neighbors
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    code = int(grid[ny, nx])
                    if code in [0, 10]:
                        near_coast = 1
                    if code == 5:
                        near_mountain = 1
                    if code == 4:
                        near_forest += 1
                    if code in [1, 2] and (dx != 0 or dy != 0):
                        n_neighbors += 1

        # Faction size
        owner = s["owner_id"]
        faction_size = sum(1 for ss in all_settlements_0 if ss["owner_id"] == owner and ss["alive"])

        features.append({
            "population": pop,
            "food": food,
            "wealth": wealth,
            "defense": defense,
            "has_port": has_port,
            "x_norm": x / w,
            "y_norm": y / h,
            "n_neighbors": n_neighbors,
            "near_coast": near_coast,
            "near_mountain": near_mountain,
            "near_forest": near_forest / 24.0,  # normalized
            "faction_size": faction_size,
            "settlement_x": x,
            "settlement_y": y,
        })
    return features


def build_dataset():
    """Build training set from all replay data."""
    X_all = []
    y_all = []
    rounds_all = []

    feature_names = None

    for round_dir in sorted(REPLAY_DIR.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        rnum = int(round_dir.name.split("_")[1])

        for seed_dir in sorted(round_dir.iterdir()):
            if not seed_dir.is_dir():
                continue

            for sample_path in sorted(seed_dir.glob("sample_*.json")):
                data = json.loads(sample_path.read_text())
                frames = data["frames"]
                frame0 = frames[0]
                frame50 = frames[50]

                grid0 = frame0["grid"]
                sett0 = frame0["settlements"]

                # Build alive set at step 50 (by position)
                alive_50 = set()
                for s in frame50["settlements"]:
                    if s["alive"]:
                        alive_50.add((s["x"], s["y"]))

                # Also check grid for settlement/port codes at step 50
                grid50 = frame50["grid"]
                for r in range(len(grid50)):
                    for c in range(len(grid50[0])):
                        if grid50[r][c] in [1, 2]:
                            alive_50.add((c, r))  # x=col, y=row

                features = extract_settlement_features(frame0, grid0, sett0)

                for feat in features:
                    sx, sy = feat.pop("settlement_x"), feat.pop("settlement_y")
                    if feature_names is None:
                        feature_names = list(feat.keys())

                    X_all.append([feat[k] for k in feature_names])
                    # Label: survived if there's a settlement/port at or near this position at step 50
                    survived = (sx, sy) in alive_50
                    y_all.append(1.0 if survived else 0.0)
                    rounds_all.append(rnum)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    rounds = np.array(rounds_all, dtype=np.int32)

    log.info(f"Dataset: {len(X)} settlements, {y.mean():.3f} survival rate")
    log.info(f"Features: {feature_names}")
    log.info(f"Rounds: {np.unique(rounds)} ({len(np.unique(rounds))} unique)")

    return X, y, rounds, feature_names


def train_loro(X, y, rounds, feature_names):
    """Leave-one-round-out cross-validation."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError:
        log.error("scikit-learn not available, trying xgboost...")
        try:
            import xgboost as xgb
            return _train_loro_xgb(X, y, rounds, feature_names)
        except ImportError:
            log.error("Neither sklearn nor xgboost available!")
            return None

    unique_rounds = sorted(np.unique(rounds))
    results = []

    for held_out in unique_rounds:
        train_mask = rounds != held_out
        test_mask = rounds == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(float)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0
        z_true = y_test.mean()
        z_pred = probs.mean()

        results.append({
            "round": held_out,
            "acc": acc,
            "auc": auc,
            "z_true": z_true,
            "z_pred": z_pred,
            "n_test": len(y_test),
        })
        log.info(f"R{held_out}: acc={acc:.3f} auc={auc:.3f} z_true={z_true:.3f} z_pred={z_pred:.3f} (n={len(y_test)})")

    # Feature importance
    log.info("\nFeature importance (last fold):")
    for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        log.info(f"  {name}: {imp:.4f}")

    return results, model


def _train_loro_xgb(X, y, rounds, feature_names):
    """XGBoost LORO fallback."""
    import xgboost as xgb

    unique_rounds = sorted(np.unique(rounds))
    results = []

    for held_out in unique_rounds:
        train_mask = rounds != held_out
        test_mask = rounds == held_out

        dtrain = xgb.DMatrix(X[train_mask], label=y[train_mask], feature_names=feature_names)
        dtest = xgb.DMatrix(X[test_mask], label=y[test_mask], feature_names=feature_names)

        params = {
            "max_depth": 5, "eta": 0.1, "objective": "binary:logistic",
            "eval_metric": "auc", "subsample": 0.8, "seed": 42,
        }
        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

        probs = model.predict(dtest)
        preds = (probs > 0.5).astype(float)
        acc = float(np.mean(preds == y[test_mask]))
        z_true = float(y[test_mask].mean())
        z_pred = float(probs.mean())

        results.append({
            "round": held_out,
            "acc": acc,
            "z_true": z_true,
            "z_pred": z_pred,
            "n_test": len(y[test_mask]),
        })
        log.info(f"R{held_out}: acc={acc:.3f} z_true={z_true:.3f} z_pred={z_pred:.3f}")

    return results, model


def main():
    X, y, rounds, feature_names = build_dataset()

    if len(X) == 0:
        log.error("No data! Ensure replays/ directory has samples.")
        return

    results, model = train_loro(X, y, rounds, feature_names)

    # Summary
    print("\n" + "=" * 60)
    print("SETTLEMENT SURVIVAL MODEL — LORO RESULTS")
    print("=" * 60)
    for r in results:
        z_err = abs(r["z_true"] - r["z_pred"])
        print(f"  R{r['round']}: acc={r['acc']:.3f}  z_true={r['z_true']:.3f}  z_pred={r['z_pred']:.3f}  z_err={z_err:.3f}")

    z_errors = [abs(r["z_true"] - r["z_pred"]) for r in results]
    print(f"\n  Mean z error: {np.mean(z_errors):.4f}")
    print(f"  Max z error:  {np.max(z_errors):.4f}")


if __name__ == "__main__":
    main()
