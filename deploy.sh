#!/bin/bash
# Deploy to new DataCrunch server and start all-data training
# Usage: bash deploy.sh <server-ip>
set -e

HOST="${1:?Usage: bash deploy.sh <server-ip>}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deploying to root@$HOST ==="

# 1. Setup server
echo ">>> Step 1: Server setup..."
ssh root@$HOST 'bash -s' < "$DIR/setup_server.sh"

# 2. Upload training script + run.py + validate_server.py
echo ""
echo ">>> Step 2: Uploading scripts..."
scp "$DIR/train_cls_alldata.py" root@$HOST:/clade/ng/
scp "$DIR/train_classifier.py" root@$HOST:/clade/ng/
scp "$DIR/validate_server.py" root@$HOST:/clade/ng/
scp "$DIR/submission/run.py" root@$HOST:/clade/ng/submission/

# 3. Upload model weights (detector.onnx + current classifier)
echo ""
echo ">>> Step 3: Uploading weights..."
scp "$DIR/submission/detector.onnx" root@$HOST:/clade/ng/submission/
scp "$DIR/submission/classifier.safetensors" root@$HOST:/clade/ng/submission/

# 4. Start training in background
echo ""
echo ">>> Step 4: Starting all-data classifier training..."
ssh root@$HOST "cd /clade/ng && source /clade/venv/bin/activate && nohup python train_cls_alldata.py > /clade/ng/train_alldata.log 2>&1 &"

echo ""
echo "=== Deploy complete ==="
echo "Monitor: ssh root@$HOST 'tail -f /clade/ng/train_alldata.log'"
echo "After training: ssh root@$HOST 'source /clade/venv/bin/activate && cd /clade/ng && python validate_server.py'"
