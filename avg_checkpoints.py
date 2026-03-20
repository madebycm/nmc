"""Average last N classifier checkpoints into one. Zero-cost accuracy boost."""
from safetensors.torch import load_file, save_file
from pathlib import Path
import torch

checkpoints = sorted(Path("/Users/xc/www/nm/checkpoints").glob("classifier_e*.safetensors"))
# Use last 5
last_n = checkpoints[-5:]
print(f"Averaging {len(last_n)} checkpoints:")
for p in last_n:
    print(f"  {p.name}")

avg_sd = {}
for i, path in enumerate(last_n):
    sd = load_file(str(path))
    for k, v in sd.items():
        v = v.float()
        if k not in avg_sd:
            avg_sd[k] = v.clone()
        else:
            avg_sd[k] += v
for k in avg_sd:
    avg_sd[k] /= len(last_n)
    avg_sd[k] = avg_sd[k].half()

out = Path("/Users/xc/www/nm/checkpoints/classifier_avg5.safetensors")
save_file(avg_sd, str(out))
print(f"Saved averaged model: {out} ({out.stat().st_size / 1e6:.1f} MB)")
