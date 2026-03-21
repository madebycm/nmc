# Bullet Verification Log

## Undocumented Platform Constraints (discovered the hard way)
- **predictions.json max size: 10MB** (NOT in docs)
- Docs only specify: 420MB ZIP, 3 weight files, 300s timeout, 8GB RAM

## Bullet #1: submission_5.4_cos20.zip — FAILED
- **Error**: predictions.json 18MB, exceeds 10MB limit
- **Root cause**: K=15 top-K × ~300 detections × ~50 images = ~225K predictions → 18MB
- **Lesson**: No output size cap in run.py

## Fix Applied (v2)
- Added `score < 0.01` early exit (scores decrease monotonically with rank decay)
- Added per-image cap: 500 predictions (5x COCO maxDets=100, safe margin)
- Tested on H100 with all 248 training images:
  - 248 images → 124,000 preds → 12MB
  - Est. 50 test images → ~25,000 preds → **~2MB** (well under 10MB)
  - Score range: 0.0149 - 0.9394
  - All images hit cap=500 (meaning we could lower to 300 if needed)

## Verified ZIPs
| File | Size | Classifier | Status |
|------|------|-----------|--------|
| submission_5.4_cos20_v2.zip | 295 MB | cos20 all-data | VERIFIED — output ~2MB for 50 images |
| submission_5.4_v2.zip | 295 MB | mixup_e3 | VERIFIED — same run.py |

## Bullet #2: submission_5.4_cos20_safe.zip — SUCCESS
- **Score: 0.9140** (rank #25, leader 0.9255)
- Delta vs v5.0: +0.0036
- Output size: within limits
- Classifier: cos20 all-data MixUp

## Remaining Bullets: 2 (resets midnight UTC)
## Bullet Usage: 1 burned (size bug) + 1 scored (0.9140) = 2 used, 1 remaining today
