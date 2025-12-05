# Resume Training Guide

## Overview

You've completed 120 epochs and want to train for 80 more epochs (120→200 total).

The training script now supports checkpoint resuming!

---

## 📋 Steps to Resume Training

### Step 1: Find Your Best Checkpoint

List your saved models:
```bash
ls -lh /ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model/
```

You should see files like:
```
gene_epoch_0_0.xxxx
gene_epoch_1_0.xxxx
...
gene_epoch_119_0.xxxx  ← Last epoch (120th epoch, 0-indexed as 119)
disc_epoch_0
disc_epoch_1
...
disc_epoch_119  ← Last discriminator
```

**Find the best model** (lowest validation loss):
- Look for the smallest number after `gene_epoch_119_`
- Example: `gene_epoch_119_0.4523` (validation loss = 0.4523)

---

### Step 2: Modify train.py Configuration

Open `/ghome/fewahab/My_5th_pap/Ab4-BSRNN/B1/train.py` and edit the `Config` class:

```python
class Config:
    # Training hyperparameters
    epochs = 200  # ← CHANGE FROM 120 TO 200 (for 80 more epochs)
    batch_size = 6
    log_interval = 500
    decay_epoch = 10
    init_lr = 1e-3
    cut_len = int(16000 * 2)
    loss_weights = [0.5, 0.5, 1]

    # Server paths
    data_dir = '/gdata/fewahab/data/VoicebanK-demand-16K'
    save_model_dir = '/ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model'

    # Resume training - SET THESE TO RESUME
    resume_training = True  # ← CHANGE TO True
    resume_epoch = 119  # ← Last completed epoch (0-indexed, so 119 = epoch 120)
    resume_generator = 'gene_epoch_119_0.xxxx'  # ← YOUR BEST MODEL
    resume_discriminator = 'disc_epoch_119'  # ← CORRESPONDING DISCRIMINATOR
```

**Example** (replace with your actual filenames):
```python
resume_training = True
resume_epoch = 119
resume_generator = 'gene_epoch_119_0.4523'  # Replace with actual filename
resume_discriminator = 'disc_epoch_119'
```

---

### Step 3: Run Training

```bash
cd /ghome/fewahab/My_5th_pap/Ab4-BSRNN/B1/
python train.py
```

---

## 📊 Expected Output

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

Scheduler adjusted: learning rate = 0.000XXX
Epoch 120, Step 1, loss: 0.xxxx, disc_loss: 0.xxxx, PESQ: x.xx
Epoch 120, Step 500, loss: 0.xxxx, disc_loss: 0.xxxx, PESQ: x.xx
...
```

Training will continue from epoch 120 to 200 (80 more epochs).

---

## 🔍 What Happens During Resume

1. **Models loaded**: Generator and discriminator weights restored
2. **Epoch counter adjusted**: Starts from 120 (not 0)
3. **Scheduler adjusted**: Learning rate matches epoch 120 state
4. **Training continues**: Epochs 120-199 (80 more epochs)
5. **New checkpoints saved**: `gene_epoch_120_*`, `gene_epoch_121_*`, etc.

---

## ⚙️ Technical Details

### Learning Rate Decay

- Your scheduler: `StepLR(step_size=10, gamma=0.98)`
- Initial LR: 0.001
- After 120 epochs: LR = 0.001 × 0.98^12 ≈ 0.000786
- The resume code automatically adjusts scheduler to this state

### Discriminator Usage

- Epochs 0-59: Generator only (no discriminator loss)
- Epochs 60-119: Generator + Discriminator
- **Epochs 120-199**: Generator + Discriminator (resumed)

The discriminator will be active since epoch ≥ 60.

---

## 📝 Quick Reference Script

Create a file `resume_train.sh`:

```bash
#!/bin/bash
# Resume training script

# Find the best model (lowest validation loss)
cd /ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model/
echo "Available checkpoints:"
ls -1 gene_epoch_119_* 2>/dev/null | head -10

echo ""
echo "Select the best checkpoint (lowest loss) and update train.py:"
echo "  resume_training = True"
echo "  resume_epoch = 119"
echo "  resume_generator = 'gene_epoch_119_0.XXXX'"
echo "  resume_discriminator = 'disc_epoch_119'"
echo ""
echo "Then run:"
echo "  cd /ghome/fewahab/My_5th_pap/Ab4-BSRNN/B1/"
echo "  python train.py"
```

Make it executable:
```bash
chmod +x resume_train.sh
./resume_train.sh
```

---

## ✅ Checklist

Before running:
- [ ] Located best checkpoint: `gene_epoch_119_*`
- [ ] Updated `epochs = 200` in Config
- [ ] Set `resume_training = True`
- [ ] Set `resume_epoch = 119`
- [ ] Set `resume_generator = 'gene_epoch_119_0.xxxx'`
- [ ] Set `resume_discriminator = 'disc_epoch_119'`
- [ ] Verified checkpoint files exist

---

## 🎯 Alternative: Continue Training WITHOUT Modifying epochs

If you want to keep `epochs=120` but train more:

**Option A: Train another 80 epochs separately**
```python
resume_training = True
resume_epoch = 119
epochs = 120  # Will train 1 more epoch (120→121)
```

Then change:
```python
epochs = 140  # Train 120→140 (20 more)
resume_epoch = 119  # Still start from 120
```

Repeat until you reach 200.

**Option B: Better - just set epochs=200 once** (Recommended)
```python
resume_training = True
resume_epoch = 119
epochs = 200  # Train 120→200 (80 more) - ONE RUN
```

---

## 🚨 Troubleshooting

### Error: "Generator checkpoint not found"
- Check the path: `/ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model/`
- Verify filename exactly matches (case-sensitive)
- Use `ls` to see actual filenames

### Error: "KeyError" or "RuntimeError: size mismatch"
- Ensure checkpoint was saved with same architecture
- Confirm `num_channel=64, num_layer=5` matches

### Training starts from epoch 0
- Verify `resume_training = True` (not False)
- Check `resume_epoch = 119` is set
- Ensure checkpoint paths are not None

---

## 📈 Monitoring Resume Training

```bash
# Watch training progress
tail -f /ghome/fewahab/My_5th_pap/Ab4-BSRNN/B1/logs/training.log

# Check new checkpoints
watch -n 10 "ls -lht /ghome/fewahab/Sun-Models/Ab-5/BSRNN/saved_model/ | head -20"
```

---

## ✅ Summary

**To resume training for 80 more epochs (120→200):**

1. Find best checkpoint: `gene_epoch_119_0.xxxx`
2. Edit train.py:
   - `epochs = 200`
   - `resume_training = True`
   - `resume_epoch = 119`
   - `resume_generator = 'gene_epoch_119_0.xxxx'`
   - `resume_discriminator = 'disc_epoch_119'`
3. Run: `python train.py`
4. Training continues from epoch 120 to 200

**Done!** Your model will train for 80 more epochs with all states properly restored. 🚀
