# Resume Training Guide

Complete guide to resume training from a saved checkpoint.

---

## Quick Start (120 → 200 epochs)

### Step 1: Find Your Best Checkpoint

```bash
cd /ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model/
ls -1 gene_epoch_119_*
```

**Choose the checkpoint with the LOWEST loss number** (smaller is better):
```
gene_epoch_119_0.4523  ← Use this (lower loss)
gene_epoch_119_0.4891  ← Not this (higher loss)
```

Also verify the discriminator checkpoint exists:
```bash
ls -1 disc_epoch_119
```

---

### Step 2: Edit train.py Configuration

Open `train.py` and modify these lines in the `Config` class:

**BEFORE:**
```python
class Config:
    epochs = 120

    resume_training = False
    resume_epoch = 0
    resume_generator = ''
    resume_discriminator = ''
```

**AFTER:**
```python
class Config:
    epochs = 200  # ← Change: new target (120 + 80 more = 200)

    resume_training = True  # ← Change: enable resume
    resume_epoch = 119  # ← Change: last completed epoch (0-indexed)
    resume_generator = 'gene_epoch_119_0.4523'  # ← Change: YOUR filename
    resume_discriminator = 'disc_epoch_119'  # ← Change: discriminator
```

**Important**: Replace `gene_epoch_119_0.4523` with YOUR actual filename from Step 1!

---

### Step 3: Run Training

```bash
cd /ghome/fewahab/My_5th_pap/Ab4-BSRNN/B1/
python train.py
```

---

## Expected Output

```
======================================================================
RESUMING FROM CHECKPOINT
======================================================================
Loading generator from: /ghome/fewahab/.../gene_epoch_119_0.4523
✓ Generator loaded successfully
Loading discriminator from: /ghome/fewahab/.../disc_epoch_119
✓ Discriminator loaded successfully
✓ Will resume from epoch 120 (continuing to epoch 200)
✓ Scheduler will skip 120 steps to match training progress
======================================================================
RESUME SUMMARY:
  Completed epochs: 0-119 (120 epochs)
  Resuming from: epoch 120
  Target epochs: 200
  Remaining epochs: 80
======================================================================

Scheduler adjusted: learning rate = 0.000786
Epoch 120, Step 1, loss: 0.xxxx, disc_loss: 0.xxxx, PESQ: x.xx
```

Training will continue from **epoch 120 to 200** (80 more epochs).

---

## What Happens During Resume

1. ✓ **Model weights restored** from checkpoint
2. ✓ **Discriminator weights restored** from checkpoint
3. ✓ **Epoch counter set** to 120 (not 0)
4. ✓ **Learning rate scheduler adjusted** to match epoch 120 state
5. ✓ **Training continues** for epochs 120-199
6. ✓ **New checkpoints saved**: `gene_epoch_120_*`, `gene_epoch_121_*`, etc.

---

## Configuration Reference

All settings are in the `Config` class at the top of `train.py`:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `epochs` | Total epochs to train | `200` (for 80 more) |
| `resume_training` | Enable/disable resume | `True` or `False` |
| `resume_epoch` | Last completed epoch (0-indexed) | `119` (= 120th epoch) |
| `resume_generator` | Generator checkpoint filename | `'gene_epoch_119_0.4523'` |
| `resume_discriminator` | Discriminator checkpoint filename | `'disc_epoch_119'` |

**Note**: Use filenames only, not full paths. The script automatically adds the path from `save_model_dir`.

---

## Error Messages and Solutions

### ❌ "resume_generator must be specified"

**Cause**: `resume_generator` is empty or not set.

**Fix**: Set the actual checkpoint filename:
```python
resume_generator = 'gene_epoch_119_0.4523'  # Not empty string ''
```

---

### ❌ "Generator checkpoint not found"

**Cause**: File doesn't exist at the expected path.

**Fix**:
1. Check the file exists:
   ```bash
   ls /ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model/gene_epoch_119_*
   ```
2. Verify `save_model_dir` is correct in `Config`
3. Use exact filename (case-sensitive, including the loss number)

---

### ❌ "Invalid resume configuration! resume_epoch >= target epochs"

**Cause**: You're trying to resume from epoch 119 but `epochs = 120` (already done).

**Fix**: Increase `epochs` to train more:
```python
epochs = 200  # Not 120
```

---

### ❌ Training starts from epoch 0 instead of 120

**Cause**: Resume not enabled.

**Fix**: Verify all settings:
```python
resume_training = True  # Must be True, not False
resume_epoch = 119  # Must be set
resume_generator = 'gene_epoch_119_0.xxxx'  # Must have filename
```

---

## Learning Rate Schedule

Your configuration uses:
```python
scheduler = StepLR(step_size=10, gamma=0.98)
init_lr = 1e-3
```

**Learning rate at different epochs:**
- Epoch 0: 0.001000
- Epoch 10: 0.000980
- Epoch 20: 0.000961
- Epoch 30: 0.000942
- ...
- Epoch 120: 0.000786 ← **Resume point**
- Epoch 130: 0.000770
- ...
- Epoch 200: 0.000634

The resume code automatically adjusts the scheduler to start at the correct learning rate for epoch 120.

---

## Verification Checklist

Before running, verify:

- [ ] You found the best checkpoint: `gene_epoch_119_*` (lowest loss)
- [ ] You updated `epochs = 200` (or your target)
- [ ] You set `resume_training = True`
- [ ] You set `resume_epoch = 119`
- [ ] You set `resume_generator = 'gene_epoch_119_0.xxxx'` (exact filename)
- [ ] You set `resume_discriminator = 'disc_epoch_119'`
- [ ] Both checkpoint files exist in `save_model_dir`
- [ ] You're in the correct directory to run `python train.py`

---

## Monitoring Training

**Watch training output:**
```bash
# If logging to file
tail -f training.log

# Or watch console output directly
```

**Check new checkpoints:**
```bash
watch -n 30 "ls -lht /ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model/ | head -20"
```

**Monitor GPU:**
```bash
watch -n 5 nvidia-smi
```

---

## Summary

**To resume training for 80 more epochs (120 → 200):**

1. Find best checkpoint: `ls gene_epoch_119_*` and pick lowest loss
2. Edit `train.py` Config:
   - `epochs = 200`
   - `resume_training = True`
   - `resume_epoch = 119`
   - `resume_generator = 'gene_epoch_119_0.xxxx'` (your actual file)
   - `resume_discriminator = 'disc_epoch_119'`
3. Run: `python train.py`
4. Training continues from epoch 120 to 200 ✓

---

**Need Help?**

If you encounter errors, the script now provides detailed error messages explaining:
- What went wrong
- What file it was looking for
- How to fix the issue

Simply read the error message and follow the instructions!
