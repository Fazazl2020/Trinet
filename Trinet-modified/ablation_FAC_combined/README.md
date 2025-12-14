# Ablation Study 3: Combined (Multiplicative + Direct Weighting)

## Combined Modification

**Applies BOTH theory-based modifications together**:
1. ✅ Multiplicative gating (from Ablation 1)
2. ✅ Direct frequency band weighting (from Ablation 2 concept)

### Theoretical Justification

Combines the strengths of both approaches:
- **Multiplicative operation**: Aligns with Wiener/IRM optimal filtering
- **Direct band weighting**: Applies frequency-specific gains without PE indirection

Tests whether both modifications provide **additive** or **synergistic** benefits.

### Architecture

**Two-Stage Processing**:

```python
# Stage 1: Multiplicative PE gating (Ablation 1)
X_pe = X * (1.0 + gate * attn * P_freq_scaled)

# Stage 2: Direct band weighting (Ablation 2 concept)
X_final = X_pe * freq_mask
```

**Final operation**: `X_out = (X * (1 + PE_gain)) * band_mask`

### Code Changes

**New Components**:
1. `DirectBandMask`: Applies direct frequency-band gains (second stage)
2. `CombinedGatingAndWeighting`: Two-stage processing module
3. `FACLayer`: Uses combined approach

**Processing Flow**:
```
Input X
  ↓
Stage 1: Multiplicative PE Gating
  - Apply learnable PE with multiplicative combination
  - Output: X_pe = X * (1 + gated_PE)
  ↓
Stage 2: Direct Band Weighting
  - Apply direct frequency-specific gains
  - Low band (0-300Hz): gain[0]
  - Mid band (300-3400Hz): gain[1]
  - High band (3400Hz+): gain[2]
  ↓
Output: X_final = X_pe * freq_mask
```

### Expected Improvement

- **Predicted PESQ gain**: +0.20 to +0.35
- **Rationale**: Combines benefits of both modifications
- **Possibility**: Synergistic effects (better than either alone)

### Comparison

| Modification | Ablation 1 | Ablation 2 | **Ablation 3** |
|--------------|------------|------------|----------------|
| Multiplicative | ✅ | ✅ (indirect) | ✅ |
| Direct weighting | ❌ | ✅ | ✅ |
| Uses PE | ✅ | ❌ | ✅ |
| Expected gain | +0.15-0.25 | +0.10-0.20 | **+0.20-0.35** |

## Usage

Replace the `module.py` file in your training directory:

```bash
cp ablation_FAC_combined/module.py /path/to/training/module.py
```

All other files (utils.py, train_improved.py, etc.) remain unchanged.

## Implementation Details

- ✅ **Syntax verified**: `python -m py_compile` passed
- ✅ **Drop-in compatible**: No training code changes needed
- ✅ **Learnable parameters**: Both PE band weights and direct gains are learned
- ✅ **Theory-based**: Combines two independently justified modifications

## Hypothesis

If the combined ablation performs **better** than either Ablation 1 or 2 alone, it suggests:
- Both issues (additive vs multiplicative AND indirect vs direct) are genuine bottlenecks
- The modifications are complementary rather than redundant

If it performs **similar** to one of the single ablations:
- One modification captures most of the improvement
- The other is less critical

## Next Steps

Run **all three ablations** in parallel:
1. Ablation 1: Multiplicative only
2. Ablation 2: Direct weighting only
3. **Ablation 3: Combined** (this one)

Compare results to determine the best approach.
