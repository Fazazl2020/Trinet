# ABLATION STUDY MODELS

This directory contains the ablation study models for the Trinet speech enhancement project.

## Directory Structure

```
ablation_models/
├── M1_Conv2D_StandardTransformer/     # Baseline: Standard components only
├── M2_FAC_StandardTransformer/        # FAC contribution test
├── M3_FAC_SingleBranchMRHA/           # Single-resolution attention baseline
└── M4_FAC_FullMRHA/                   # Full proposed model
```

## Model Descriptions

### M1: Standard Baseline (Conv2D + Standard Transformer)
**Purpose:** Establish baseline performance without any novel components

**Architecture:**
- Encoder: Standard Conv2D (NO frequency-adaptive processing)
- Bottleneck: Standard Multi-Head Self-Attention
- Decoder: Standard ConvTranspose2D

**Expected Performance:** PESQ ~2.7-2.8

**Novel Components:** NONE

---

### M2: FAC + Standard Transformer
**Purpose:** Isolate the contribution of Frequency-Adaptive Convolution

**Architecture:**
- Encoder: FAC layers (NOVEL - frequency bands, gating, attention)
- Bottleneck: Standard Multi-Head Self-Attention
- Decoder: Standard ConvTranspose2D

**Expected Performance:** PESQ ~2.85-2.95

**Novel Components:**
- ✓ AdaptiveFrequencyBandPositionalEncoding
- ✓ DepthwiseFrequencyAttention
- ✓ GatedPositionalEncoding
- ✓ FACLayer
- ✗ Multi-Resolution Attention
- ✗ Hybrid Attention Branches
- ✗ Adaptive Gating

---

### M3: FAC + Single-Branch MRHA
**Purpose:** Establish single-resolution attention baseline for comparing multi-resolution benefit

**Architecture:**
- Encoder: FAC layers (NOVEL)
- Bottleneck: AIA_Transformer with ONLY dot-product attention (NO multi-resolution, NO cosine, NO gating)
- Decoder: Standard ConvTranspose2D

**Expected Performance:** PESQ ~2.9-3.0

**Novel Components:**
- ✓ Full FAC (same as M2)
- ✓ Row/Column factorized attention
- ✓ Sinusoidal Positional Encoding
- ✗ Cross-Resolution Branch
- ✗ Cosine Attention Branch
- ✗ 3-Way Learnable Gating

**Key Difference from M2:** Uses factorized row/column attention (same structure as full AIA) but without multi-resolution/hybrid branches

---

### M4: FAC + Full MRHA (Proposed Model)
**Purpose:** Complete proposed model with all novel components

**Architecture:**
- Encoder: FAC layers (NOVEL)
- Bottleneck: Full AIA_Transformer with Hybrid_SelfAttention_MRHA3 (NOVEL)
- Decoder: Standard ConvTranspose2D

**Expected Performance:** PESQ ~3.0-3.1

**Novel Components:**
- ✓ Full FAC
- ✓ Cross-Resolution Attention (global context)
- ✓ Local Dot-Product Attention
- ✓ Cosine Attention with Learnable Temperature
- ✓ 3-Way Learnable Gating
- ✓ Row/Column Factorization
- ✓ Sinusoidal Positional Encoding

---

## Training Instructions

### Step 1: Navigate to Model Directory

```bash
cd /home/user/Trinet/Trinet-modified/ablation_models/M1_Conv2D_StandardTransformer
# (or M2, M3, M4)
```

### Step 2: Copy Training Files

You need to copy these files to each ablation directory:
- `train.py` (modify `save_model_dir` for each model)
- `dataloader.py`
- `utils.py`
- `evaluation.py`

```bash
# Example for M1:
cp ../../train.py ./
cp ../../dataloader.py ./
cp ../../utils.py ./
cp ../../evaluation.py ./
```

### Step 3: Modify train.py

In each model's `train.py`, change the save directory:

```python
# M1
save_model_dir = '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M1_Baseline/ckpt'

# M2
save_model_dir = '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M2_FAC/ckpt'

# M3
save_model_dir = '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M3_FAC_SingleMRHA/ckpt'

# M4
save_model_dir = '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M4_FAC_FullMRHA/ckpt'
```

### Step 4: Run Training

```bash
python train.py
```

### Step 5: Monitor Training

All models should train for 120 epochs with same hyperparameters for fair comparison.

---

## Validation Script

Before training, validate that each model works:

```bash
cd /home/user/Trinet/Trinet-modified/ablation_models
python validate_models.py
```

This will test:
- Module imports correctly
- Model instantiates without errors
- Forward pass works
- Output shapes are correct
- No NaN/Inf values

---

## Expected Ablation Results

| Model | PESQ ↑ | Δ from M1 | Key Finding |
|-------|--------|-----------|-------------|
| **M1: Baseline** | 2.75 | - | Standard components baseline |
| **M2: +FAC** | 2.90 | +0.15 | FAC improves spectral modeling |
| **M3: +Single MRHA** | 2.98 | +0.23 | Factorized attention helps |
| **M4: +Full MRHA** | 3.06 | +0.31 | Multi-resolution adds 0.08 |

**Key Insights:**
- **M2-M1 = 0.15:** FAC contribution (largest single gain)
- **M3-M2 = 0.08:** Row/column factorization benefit
- **M4-M3 = 0.08:** Multi-resolution + hybrid + gating benefit
- **Total = 0.31:** Complete system improvement

---

## Notes

1. **Same hyperparameters:** All models use identical training config for fair comparison
2. **Same data:** All models train on same VoiceBank+DEMAND splits
3. **Same capacity:** All models use `num_channel=64, num_layer=5`
4. **Same discriminator:** All models use identical MetricGAN discriminator

Only the encoder (Conv2D vs FAC) and bottleneck (Transformer variants) differ.

---

## File Requirements

Each model directory needs:
- `module.py` ✓ (already included)
- `train.py` (copy and modify)
- `dataloader.py` (copy)
- `utils.py` (copy)
- `evaluation.py` (copy)

---

## Troubleshooting

### Import Error
```python
ModuleNotFoundError: No module named 'utils'
```
**Solution:** Copy `utils.py` to the model directory

### Shape Mismatch
All models use adaptive shape matching - this should not occur. If it does, check that you copied the correct `module.py`.

### NaN Loss
This can happen if learning rate is too high. All models should use `init_lr=1e-3` as in original.

---

## Citation

If you use these ablation models, cite:

```bibtex
@article{trinet2024,
  title={Trinet: Frequency-Adaptive Speech Enhancement with Multi-Resolution Hybrid Attention},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```
