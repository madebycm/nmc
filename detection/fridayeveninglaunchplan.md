# Friday Evening Launch Plan — ArcFace Classifier Retraining
**Time:** ~20:30 → midnight UTC (3.5 hrs)
**Server:** H200 141GB @ XXx--xx-H200 (ONLY)
**Goal:** Ship ArcFace-trained classifier with better embeddings for tonight's submission

---

## TODO

### Phase 1: H200 Setup (30 min)
- [ ] Install PyTorch + CUDA on H200
- [ ] Install timm, safetensors, pycocotools, pillow
- [ ] Transfer training data (coco_dataset + product_images) to H200
- [ ] Transfer current classifier weights (EVA-02 Base pretrained) to H200
- [ ] Verify GPU training works (quick smoke test)

### Phase 2: OOF Baseline (30 min)
- [ ] Train EVA-02 Base with CE-only on 198 train images, eval on 50 val
- [ ] Record honest val cls_mAP (uncontaminated baseline)
- [ ] Identify per-class accuracy — which of the 84 rare classes fail?

### Phase 3: ArcFace Training (45 min)
- [ ] Write ArcFace training script: CE + ArcFace head, balanced sampler
- [ ] Oversample 84 rare classes (≤5 examples) 10x
- [ ] Train fold-1 on 198 train, eval on 50 val — honest comparison vs CE-only
- [ ] If ArcFace wins: retrain on ALL 248 images with ArcFace (production weights)

### Phase 4: Regenerate Embeddings (15 min)
- [ ] Extract embeddings from ArcFace model for all 6,606 references
- [ ] L2-normalize, save as ref_embeddings_finetuned.npy
- [ ] Verify shape (6606×768 FP16)

### Phase 5: Validate + Package (30 min)
- [ ] Run full pipeline on H200: detection + ArcFace classifier + kNN
- [ ] Compare det/cls/blend vs v4.3 baseline
- [ ] Download new weights to local
- [ ] Package submission ZIP
- [ ] Sanity check: file sizes, runtime estimate

### Phase 6: Submit (buffer)
- [ ] Final review
- [ ] Submit before midnight UTC

---

## Key Decisions

**ArcFace config:**
- Margin: 0.5 (standard)
- Scale: 64 (standard)
- Balanced sampler: oversample rare classes to min 20 examples per class
- Epochs: 30 (with early stopping on fold val)
- LR: 1e-4 (cosine decay)
- Backbone: EVA-02 Base (same as current, drop-in replacement)

**What we're betting on:**
ArcFace loss produces embeddings where cosine similarity is geometrically meaningful. This directly improves kNN retrieval on rare classes without routing hacks. The supervised CE head still handles confident predictions; the ArcFace-trained embedding space rescues the uncertain ones.

**Abort criteria:**
If fold-1 ArcFace shows no improvement over CE-only on rare classes → submit v4.3 as-is and use tomorrow's 3 bullets for iteration.

---

## Files to Produce
| File | Source | Destination |
|------|--------|-------------|
| classifier_arcface.safetensors | H200 training | submission/ |
| ref_embeddings_finetuned.npy | H200 embedding extraction | submission/ |
| run.py | Keep v4.3 (kNN routing compatible) | submission/ |
