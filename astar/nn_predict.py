"""Neural net inference for Astar Island predictions.

Multi-model ensemble: v2 + v3 + v4 (Conditional U-Net with Global Context Vector).
All with TTA (8x rotation/flip augmentation).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
CONTEXT_DIM = 8

# v2/v3 use 13 channels (12 spatial + 1 z broadcast)
IN_CHANNELS_V2V3 = 13
# v4 uses 20 channels (12 spatial + 8 context broadcast)
SPATIAL_CHANNELS = 12
IN_CHANNELS_V4 = SPATIAL_CHANNELS + CONTEXT_DIM

_models = {}  # lazy cache


# ── Model architectures ──────────────────────────────────────────────


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


class ResBlockV3(nn.Module):
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


class AstarNet(nn.Module):
    """V2 architecture: 128 hidden, 6 ResBlocks."""
    def __init__(self, in_ch=IN_CHANNELS_V2V3, hidden=128, num_classes=NUM_CLASSES):
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


class AstarNetV3(nn.Module):
    """V3 architecture: 192 hidden, 8 ResBlocks, multi-scale."""
    def __init__(self, in_ch=IN_CHANNELS_V2V3, hidden=192, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlockV3(hidden, dilation=1),
            ResBlockV3(hidden, dilation=2),
            ResBlockV3(hidden, dilation=4),
            ResBlockV3(hidden, dilation=8),
            ResBlockV3(hidden, dilation=16),
            ResBlockV3(hidden, dilation=8),
            ResBlockV3(hidden, dilation=4),
            ResBlockV3(hidden, dilation=1),
        )
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


class ConditionalUNet(nn.Module):
    """V4: U-Net conditioned on 8-dim global context vector."""
    def __init__(self, in_ch=IN_CHANNELS_V4, hidden=160, num_classes=NUM_CLASSES):
        super().__init__()
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
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(),
            ResBlock(hidden * 2, dilation=1),
            ResBlock(hidden * 2, dilation=2),
        )
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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1).permute(0, 2, 3, 1)


# ── Feature encoding ─────────────────────────────────────────────────


def encode_grid_v2v3(grid: np.ndarray, z: float) -> np.ndarray:
    """Encode for v2/v3: match training channel order exactly.

    Training order: terrain(8) + distance(1) + coastal(1) + is_land(1) + z(1) + density(1) = 13ch
    """
    h, w = grid.shape
    channels = []
    # 8 one-hot terrain channels
    for code in TERRAIN_CODES:
        channels.append((grid == code).astype(np.float32))
    # Distance to nearest settlement/port (ch 8)
    positions = list(zip(*np.where(np.isin(grid, [1, 2]))))
    dist = np.full((h, w), 40.0, dtype=np.float32)
    if positions:
        for sy, sx in positions:
            yy = np.abs(np.arange(h) - sy)[:, None].astype(np.float32)
            xx = np.abs(np.arange(w) - sx)[None, :].astype(np.float32)
            dist = np.minimum(dist, yy + xx)
    channels.append(dist / 40.0)
    # Coastal adjacency (ch 9)
    ocean = (grid == 10)
    coastal = np.zeros((h, w), dtype=np.float32)
    if h > 1:
        coastal[1:] = np.maximum(coastal[1:], ocean[:-1])
        coastal[:-1] = np.maximum(coastal[:-1], ocean[1:])
    if w > 1:
        coastal[:, 1:] = np.maximum(coastal[:, 1:], ocean[:, :-1])
        coastal[:, :-1] = np.maximum(coastal[:, :-1], ocean[:, 1:])
    channels.append(coastal)
    # Is land (ch 10)
    channels.append((grid != 10).astype(np.float32))
    # Z broadcast (ch 11) — MUST come before density to match training order
    channels.append(np.full((h, w), z, dtype=np.float32))
    # Settlement density 5x5 (ch 12)
    settle = np.isin(grid, [1, 2]).astype(np.float32)
    padded = np.pad(settle, 2, mode="constant")
    density = np.zeros((h, w), dtype=np.float32)
    for dy in range(5):
        for dx in range(5):
            density += padded[dy : dy + h, dx : dx + w]
    channels.append(density / 25.0)
    return np.stack(channels)


def encode_grid_v4(grid: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Encode for v4: match v4 training channel order.

    v4 training order: spatial(12) + context(8) = 20ch
    spatial = terrain(8) + distance(1) + coastal(1) + is_land(1) + density(1)
    """
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
    # Settlement density (ch 11) — BEFORE context to match training
    settle = np.isin(grid, [1, 2]).astype(np.float32)
    padded = np.pad(settle, 2, mode="constant")
    density = np.zeros((h, w), dtype=np.float32)
    for dy in range(5):
        for dx in range(5):
            density += padded[dy : dy + h, dx : dx + w]
    channels.append(density / 25.0)
    # Context vector broadcast (ch 12-19)
    ctx_broadcast = context[:, None, None] * np.ones((CONTEXT_DIM, h, w), dtype=np.float32)
    for i in range(CONTEXT_DIM):
        channels.append(ctx_broadcast[i])
    return np.stack(channels)


# ── Model loading ────────────────────────────────────────────────────


MODEL_SPECS = {
    "v2": ("astar_nn.pt", AstarNet),
    "v3": ("astar_nn_v3.pt", AstarNetV3),
    "v4": ("astar_nn_v4.pt", ConditionalUNet),
    "replay": ("astar_nn_replay.pt", AstarNet),  # Phase 4: replay-augmented, same arch as v2
}


def _load_model(name: str):
    if name in _models:
        return _models[name]
    if name not in MODEL_SPECS:
        return None
    filename, cls = MODEL_SPECS[name]
    path = MODEL_DIR / filename
    if not path.exists():
        log.warning(f"Model {name} not found at {path}")
        return None
    model = cls()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    _models[name] = model
    log.info(f"Loaded NN model {name} from {path} on {DEVICE}")
    return model


# ── Inference ────────────────────────────────────────────────────────


def _predict_single_v2v3(model, grid: np.ndarray, z: float) -> np.ndarray:
    features = encode_grid_v2v3(grid, z)
    x = torch.tensor(features[None], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred = F.softmax(logits, dim=-1)
    return pred[0].cpu().numpy()


def _predict_single_v4(model, grid: np.ndarray, context: np.ndarray) -> np.ndarray:
    features = encode_grid_v4(grid, context)
    x = torch.tensor(features[None], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred = F.softmax(logits, dim=-1)
    return pred[0].cpu().numpy()


def _predict_with_tta_v2v3(model, grid: np.ndarray, z: float) -> np.ndarray:
    """8x TTA for v2/v3."""
    preds = []
    for rot in range(4):
        for flip in [False, True]:
            g = np.rot90(grid, rot).copy()
            if flip:
                g = np.fliplr(g).copy()
            p = _predict_single_v2v3(model, g, z)
            if flip:
                p = np.fliplr(p)
            p = np.rot90(p, -rot, axes=(0, 1))
            preds.append(p)
    return np.mean(preds, axis=0)


def _predict_with_tta_v4(model, grid: np.ndarray, context: np.ndarray) -> np.ndarray:
    """8x TTA for v4 (context vector is rotation-invariant)."""
    preds = []
    for rot in range(4):
        for flip in [False, True]:
            g = np.rot90(grid, rot).copy()
            if flip:
                g = np.fliplr(g).copy()
            p = _predict_single_v4(model, g, context)
            if flip:
                p = np.fliplr(p)
            p = np.rot90(p, -rot, axes=(0, 1))
            preds.append(p)
    return np.mean(preds, axis=0)


def predict(
    initial_grid: list[list[int]],
    z: float = 0.283,
    context: np.ndarray | None = None,
    tta: bool = True,
) -> np.ndarray | None:
    """Multi-model ensemble prediction with TTA.

    Uses v2/v3 blended by z-adaptive weights.
    Z is clipped to training support range to prevent OOD disasters.
    On healthy rounds (z>0.45), z-TTA averages across small band for robustness.
    """
    grid = np.array(initial_grid, dtype=np.int32)

    # Z-clip: prevent OOD disaster (R12 was z=0.638, outside training support)
    Z_TRAIN_MIN, Z_TRAIN_MAX = 0.018, 0.599
    z_raw = z
    z = float(np.clip(z, Z_TRAIN_MIN, Z_TRAIN_MAX))
    if z != z_raw:
        log.warning(f"Z clipped: {z_raw:.3f} -> {z:.3f} (training range [{Z_TRAIN_MIN}, {Z_TRAIN_MAX}])")

    # Regime-adaptive v2:v3 ratio — v2 more stable OOD on healthy rounds
    if z_raw > 0.40:
        v2_w, v3_w = 0.25, 0.60
        log.info(f"Healthy regime (z={z_raw:.3f}): boosted v2 ratio 0.25:0.60")
    else:
        v2_w, v3_w = 0.15, 0.70

    preds = []
    weights = []

    # Determine z values for prediction (z-TTA on healthy rounds)
    if z_raw > 0.45:
        z_band = [z, max(z - 0.03, Z_TRAIN_MIN), min(z + 0.03, Z_TRAIN_MAX)]
        log.info(f"Z-TTA: averaging over {[f'{zz:.3f}' for zz in z_band]}")
    else:
        z_band = [z]

    # V2: stable backbone
    model_v2 = _load_model("v2")
    if model_v2 is not None:
        v2_preds = []
        for zz in z_band:
            p = _predict_with_tta_v2v3(model_v2, grid, zz) if tta else _predict_single_v2v3(model_v2, grid, zz)
            v2_preds.append(p)
        preds.append(np.mean(v2_preds, axis=0))
        weights.append(v2_w)

    # V3: strongest expert
    model_v3 = _load_model("v3")
    if model_v3 is not None:
        v3_preds = []
        for zz in z_band:
            p = _predict_with_tta_v2v3(model_v3, grid, zz) if tta else _predict_single_v2v3(model_v3, grid, zz)
            v3_preds.append(p)
        preds.append(np.mean(v3_preds, axis=0))
        weights.append(v3_w)

    if not preds:
        return None

    weights = np.array(weights) / sum(weights)
    avg = sum(w * p for w, p in zip(weights, preds))
    avg = np.clip(avg, 0.003, None)
    return avg / avg.sum(axis=-1, keepdims=True)
