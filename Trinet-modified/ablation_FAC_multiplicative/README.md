# Ablation Study 1: Multiplicative Gating

## Theory-Based Modification

**Change**: GatedPositionalEncoding uses **MULTIPLICATIVE** instead of **ADDITIVE** operation

### Theoretical Justification

Based on classical speech enhancement theory:
- **Wiener Filter**: `G(ω) = S²(ω) / (S²(ω) + N²(ω))` - multiplicative gain
- **Ideal Ratio Mask (IRM)**: `M(ω) = √(S²/N²) / (1 + √(S²/N²))` - multiplicative mask
- **MMSE-STSA**: Uses multiplicative spectral gain

### Code Change

**Location**: `GatedPositionalEncoding.forward()` (line ~157)

**Original**:
```python
return X + gate * attn * P_freq_scaled  # Additive bias
```

**Modified**:
```python
return X * (1.0 + gate * attn * P_freq_scaled)  # Multiplicative gain
```

### Expected Improvement

- **Predicted PESQ gain**: +0.15 to +0.25
- **Rationale**: Aligns with classical optimal filtering theory
- **Strong theoretical basis**: Wiener/IRM approaches proven optimal under certain assumptions

## Usage

Replace the `module.py` file in your training directory with this version:

```bash
cp ablation_FAC_multiplicative/module.py /path/to/training/module.py
```

All other files (utils.py, train.py, etc.) remain unchanged.

## Changes Summary

- ✅ Modified: `GatedPositionalEncoding.forward()` - multiplicative instead of additive
- ✅ Preserved: All other components (FAC, AIA_Transformer, network architecture)
- ✅ Compatible: Drop-in replacement, no training code changes needed
