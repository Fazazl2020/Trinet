# Ablation Study 2: Direct Frequency Band Weighting

## Theory-Based Modification

**Change**: Apply band weights **DIRECTLY** to signal instead of via positional encoding

### Theoretical Justification

Classical masking approaches use direct signal weighting:
- **Ideal Ratio Mask (IRM)**: `Enhanced = Noisy × Mask(ω)` - direct multiplication
- **Spectral Subtraction**: Direct frequency-dependent operation
- **Time-Frequency Masking**: Direct per-bin gains

Current approach applies band weights to PE, which then affects signal indirectly through addition. Direct weighting is more aligned with classical theory.

### Code Changes

**Modified Classes**:

1. **Removed**: `AdaptiveFrequencyBandPositionalEncoding` - no longer used
2. **Added**: `DirectBandWeighting` - new class for direct signal weighting
3. **Modified**: `FACLayer` - uses `DirectBandWeighting` instead of `GatedPositionalEncoding`

**New Implementation**:
```python
class DirectBandWeighting(nn.Module):
    """Apply band-specific gains directly to signal"""

    def forward(self, X):
        # Create frequency-dependent gain mask
        freq_mask = torch.ones(freq_bins)
        freq_mask[:low_end] = gains[0]      # 0-300Hz
        freq_mask[low_end:mid_end] = gains[1]  # 300-3400Hz
        freq_mask[mid_end:] = gains[2]      # 3400Hz+

        # Direct multiplicative weighting
        return X * (freq_mask * gate * attn)
```

### Expected Improvement

- **Predicted PESQ gain**: +0.10 to +0.20
- **Rationale**: More direct signal processing, aligned with masking theory
- **Moderate theoretical basis**: Removes unnecessary PE layer, applies weights directly

## Usage

Replace the `module.py` file in your training directory with this version:

```bash
cp ablation_FAC_direct_weighting/module.py /path/to/training/module.py
```

All other files (utils.py, train.py, etc.) remain unchanged.

## Changes Summary

- ✅ Removed: `AdaptiveFrequencyBandPositionalEncoding` (no PE in forward pass)
- ✅ Added: `DirectBandWeighting` - direct frequency-band gains
- ✅ Modified: Band weights now directly multiply signal instead of PE
- ✅ Preserved: AIA_Transformer, network architecture, same band definitions
- ✅ Compatible: Drop-in replacement, no training code changes needed
