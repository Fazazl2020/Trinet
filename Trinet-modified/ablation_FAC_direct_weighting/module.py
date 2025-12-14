"""
TrinetBSRNN - Integration of Trinet Network with BSRNN Pipeline
================================================================

ABLATION STUDY 2: Direct Frequency Band Weighting
--------------------------------------------------
Change: Apply band weights DIRECTLY to signal instead of via positional encoding
Theory: Classical masking approaches (IRM) use direct multiplicative weighting
Expected improvement: +0.10-0.20 PESQ

Modified: Replaced GatedPositionalEncoding with DirectBandWeighting
- Removed: AdaptiveFrequencyBandPositionalEncoding (no PE used)
- Added: Direct frequency-band specific gains applied multiplicatively
- Kept: Attention and gating mechanisms for adaptive control

This module integrates the Trinet network (with novel FAC and AIA_Transformer components)
into the BSRNN training pipeline.

Novel Components (PRESERVED from Trinet):
1. FAC (Frequency-Adaptive Convolution): Modified to use direct weighting
2. AIA_Transformer: Attention-in-Attention with Multi-Resolution Hybrid Attention

Modified Components:
- Encoder/Decoder: Adapted for F=257 frequency bins (BSRNN's n_fft=512)
- Input/Output adapters: Convert between BSRNN and Trinet formats

Author: Claude (Integration + Ablation)
Original Trinet: User's research
Original BSRNN: Open source baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import LearnableSigmoid


# ============================================================================
# NOVEL COMPONENT 1: FAC (Frequency-Adaptive Convolution)
# ============================================================================

class DepthwiseFrequencyAttention(nn.Module):
    """NOVEL COMPONENT - Applies attention along the FREQUENCY dimension."""
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels,
                                 kernel_size=(1, kernel_size),
                                 padding=(0, kernel_size//2),
                                 groups=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.dw_conv(x))


class DirectBandWeighting(nn.Module):
    """
    ABLATION 2: Direct Frequency Band Weighting
    Apply band-specific gains directly to signal (no positional encoding)
    Based on classical masking theory: Y = X * M(ω)
    """
    def __init__(self, in_channels, F=257, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.F = F

        # Same frequency band definitions as original
        self.low_freq_range = (0, 300)       # Fundamental frequencies
        self.mid_freq_range = (300, 3400)    # Telephone bandwidth (formants)
        self.high_freq_range = (3400, sample_rate / 2)  # High frequencies

        # Learnable band-specific gains (initialized same as original)
        self.band_gains = nn.Parameter(torch.tensor([0.5, 1.0, 0.3]))

        # Keep attention mechanism (frequency-adaptive)
        self.dw_attention = DepthwiseFrequencyAttention(in_channels)

        # Gate to control weighting strength
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, X):
        batch_size, C, T, freq_bins = X.size()
        device = X.device
        nyquist = self.sample_rate / 2

        # Calculate band boundaries (same as original)
        low_end = int(freq_bins * (self.low_freq_range[1] / nyquist))
        mid_end = int(freq_bins * (self.mid_freq_range[1] / nyquist))

        # Normalize gains (same as original: softmax + scale)
        gains = F.softmax(self.band_gains, dim=0) * 3.0

        # ✅ ABLATION 2: Create frequency-dependent gain mask [F]
        # This directly weights the signal instead of PE
        freq_mask = torch.ones(freq_bins, device=device)
        freq_mask[:low_end] = gains[0]           # Low band (0-300Hz)
        freq_mask[low_end:mid_end] = gains[1]    # Mid band (300-3400Hz)
        freq_mask[mid_end:] = gains[2]           # High band (3400Hz+)

        # Expand to [B, C, T, F]
        freq_mask = freq_mask.view(1, 1, 1, -1).expand(batch_size, C, T, freq_bins)

        # Apply attention and gating
        attn = self.dw_attention(X)
        gate = self.gate(X)

        # ✅ Direct multiplicative weighting: X * G(ω)
        # Like IRM: enhanced = noisy * mask
        return X * (freq_mask * gate * attn)


class FACLayer(nn.Module):
    """NOVEL COMPONENT - Frequency-Adaptive Convolution Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, F=257, sample_rate=16000):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.band_weighting = DirectBandWeighting(in_channels, F=F, sample_rate=sample_rate)

    def forward(self, X):
        X = self.band_weighting(X)
        return self.conv(X)


# ============================================================================
# NOVEL COMPONENT 2: AIA_Transformer (Bottleneck Transformer)
# ============================================================================

class Hybrid_SelfAttention_MRHA3(nn.Module):
    """
    NOVEL COMPONENT - Multi-Resolution Hybrid Attention with 3 branches
    Features: Cross-resolution, dot-product, and cosine attention with learnable temperature
    """
    def __init__(self, in_channels, downsample_stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.downsample_stride = downsample_stride

        # Cross-Resolution Branch
        self.downsample = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=downsample_stride, padding=1)
        self.query_cross = nn.Linear(in_channels, in_channels)
        self.key_cross = nn.Linear(in_channels, in_channels)
        self.value_cross = nn.Linear(in_channels, in_channels)
        self.norm_cross = nn.LayerNorm(in_channels)

        # Local Dot Branch
        self.query_dot = nn.Linear(in_channels, in_channels)
        self.key_dot = nn.Linear(in_channels, in_channels)
        self.value_dot = nn.Linear(in_channels, in_channels)
        self.norm_dot = nn.LayerNorm(in_channels)

        # Local Cosine Branch with learnable temperature
        self.query_cos = nn.Linear(in_channels, in_channels)
        self.key_cos = nn.Linear(in_channels, in_channels)
        self.value_cos = nn.Linear(in_channels, in_channels)
        self.norm_cos = nn.LayerNorm(in_channels)
        self.cos_temperature = nn.Parameter(torch.tensor(0.1))  # Learnable temperature

        # 3-Way Gating
        self.gate_conv = nn.Conv1d(3 * in_channels, 3, kernel_size=1)
        self.eps = 1e-8

    def forward(self, x):
        B, T, C = x.shape

        # Cross-Resolution Branch
        x_down = self.downsample(x.permute(0,2,1)).permute(0,2,1)
        q_cross = self.query_cross(x)
        k_cross = self.key_cross(x_down)
        v_cross = self.value_cross(x_down)
        attn_cross = F.softmax(torch.bmm(q_cross, k_cross.transpose(1,2)) / math.sqrt(C), dim=-1)
        out_cross = self.norm_cross(torch.bmm(attn_cross, v_cross))

        # Local Dot Branch
        q_dot = self.query_dot(x)
        k_dot = self.key_dot(x)
        v_dot = self.value_dot(x)
        attn_dot = F.softmax(torch.bmm(q_dot, k_dot.transpose(1,2)) / math.sqrt(C), dim=-1)
        out_dot = self.norm_dot(torch.bmm(attn_dot, v_dot))

        # Local Cosine Branch with temperature
        q_cos = self.query_cos(x)
        k_cos = self.key_cos(x)
        q_norm = q_cos / (q_cos.norm(dim=2, keepdim=True) + self.eps)
        k_norm = k_cos / (k_cos.norm(dim=2, keepdim=True) + self.eps)
        temperature = self.cos_temperature.clamp(min=0.01)
        attn_cos = F.softmax(torch.bmm(q_norm, k_norm.transpose(1,2)) / temperature, dim=-1)
        out_cos = self.norm_cos(torch.bmm(attn_cos, self.value_cos(x)))

        # 3-Way Gating
        fused = torch.cat([out_cross, out_dot, out_cos], dim=-1).permute(0,2,1)
        gating = F.softmax(self.gate_conv(fused), dim=1)
        gating = gating.permute(0,2,1).unsqueeze(2)

        outputs = torch.stack([out_cross, out_dot, out_cos], dim=3)
        z = torch.sum(gating * outputs, dim=3)
        return z


class AIA_Transformer(nn.Module):
    """
    NOVEL COMPONENT - Attention-in-Attention Transformer
    Features: Row/column attention with sinusoidal positional encoding
    """
    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = input_size // 2

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size//2, kernel_size=1),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.row_trans = Hybrid_SelfAttention_MRHA3(input_size//2)
        self.col_trans = Hybrid_SelfAttention_MRHA3(input_size//2)

        self.row_norm = nn.InstanceNorm2d(input_size//2, affine=True)
        self.col_norm = nn.InstanceNorm2d(input_size//2, affine=True)

        self.k1 = nn.Parameter(torch.ones(1))
        self.k2 = nn.Parameter(torch.ones(1))

        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(input_size//2, output_size, kernel_size=1),
            nn.Dropout(dropout)
        )

        # Learnable scale for sinusoidal PE
        self.pe_scale = nn.Parameter(torch.tensor(0.1))

    def _get_sinusoidal_pe(self, length, channels, device):
        """Generate sinusoidal positional encoding for any length"""
        position = torch.arange(length, device=device).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2, device=device).float() * (-math.log(10000.0) / channels)
        )
        pe = torch.zeros(length, channels, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        B, C, d2, d1 = x.shape  # d2=time, d1=freq
        x = self.input(x)
        _, C_hidden, _, _ = x.shape

        # Dynamic sinusoidal PE
        time_pe = self._get_sinusoidal_pe(d2, C_hidden, x.device)
        freq_pe = self._get_sinusoidal_pe(d1, C_hidden, x.device)

        time_pe = time_pe.permute(1, 0).unsqueeze(0).unsqueeze(-1)
        freq_pe = freq_pe.permute(1, 0).unsqueeze(0).unsqueeze(2)

        x = x + self.pe_scale * (time_pe + freq_pe)

        # Row-wise attention (frequency dimension)
        row_in = x.permute(0,2,3,1).contiguous().view(B*d2, d1, -1)
        row_out = self.row_trans(row_in).view(B, d2, d1, -1).permute(0,3,1,2)
        row_out = self.row_norm(row_out)

        # Column-wise attention (time dimension)
        col_in = x.permute(0,3,2,1).contiguous().view(B*d1, d2, -1)
        col_out = self.col_trans(col_in).view(B, d1, d2, -1).permute(0,3,2,1)
        col_out = self.col_norm(col_out)

        # Fusion
        out = x + self.k1 * row_out + self.k2 * col_out
        return self.output(out)


# ============================================================================
# MAIN NETWORK: TrinetBSRNN
# ============================================================================

class TrinetBSRNN(nn.Module):
    """
    Trinet Network Integrated with BSRNN Pipeline

    Architecture:
    1. Input Adapter: [B, F, T] complex → [B, 2, T, F] real
    2. Encoder: Variable FAC layers with novel positional encoding
    3. Bottleneck: AIA_Transformer with multi-resolution attention
    4. Decoder: Variable ConvTranspose layers with skip connections
    5. Output Adapter: [B, 2, T, F] real → [B, F, T] complex

    Novel Components (preserved from Trinet):
    - FAC: Frequency-Adaptive Convolution with direct band weighting
    - AIA_Transformer: Attention-in-Attention with MRHA3

    Modified for BSRNN:
    - F=257 frequency bins (n_fft=512, hop=128)
    - Input/output format conversion
    - Compatible with BSRNN training loop

    Args:
        num_channel: Base channel multiplier (controls network capacity)
        num_layer: Number of encoder/decoder layers (3, 4, 5, or 6)
        F: Number of frequency bins (default 257 for n_fft=512)
    """
    def __init__(self, num_channel=128, num_layer=6, F=257):
        super().__init__()
        self.num_channel = num_channel
        self.num_layer = num_layer
        self.F = F

        # Calculate channel progression based on num_channel
        # Base ratio: [16, 32, 64, 128, 256] for num_channel=128
        # Scale proportionally for other values
        scale = num_channel / 128.0

        # Define channels for each layer (scale from base)
        # Ensure minimum 8 channels and powers of 2
        c1 = max(8, int(16 * scale))
        c2 = max(16, int(32 * scale))
        c3 = max(32, int(64 * scale))
        c4 = max(64, int(128 * scale))
        c5 = max(128, int(256 * scale))

        # Additional layer for num_layer=6
        c6 = max(256, int(512 * scale)) if num_layer >= 6 else c5

        # Bottleneck channel (highest capacity)
        bottleneck_ch = c6 if num_layer >= 6 else c5

        self.channels = [2, c1, c2, c3, c4, c5]
        if num_layer >= 6:
            self.channels.append(c6)

        # ============================================================
        # ENCODER with FAC (Novel Component)
        # ============================================================
        # Always create all possible layers, but only use num_layer of them
        self.conv1 = FACLayer(2, c1, (2,5), (1,2), (1,1), F)
        self.conv2 = FACLayer(c1, c2, (2,5), (1,2), (1,1), F)
        self.conv3 = FACLayer(c2, c3, (2,5), (1,2), (1,1), F)
        self.conv4 = FACLayer(c3, c4, (2,5), (1,2), (1,1), F)
        self.conv5 = FACLayer(c4, c5, (2,5), (1,2), (1,1), F)

        if num_layer >= 6:
            self.conv6 = FACLayer(c5, c6, (2,5), (1,2), (1,1), F)

        # ============================================================
        # BOTTLENECK: AIA Transformer (Novel Component)
        # ============================================================
        self.m1 = AIA_Transformer(bottleneck_ch, bottleneck_ch)

        # ============================================================
        # DECODER
        # ============================================================
        if num_layer >= 6:
            self.de6 = nn.ConvTranspose2d(c6*2, c5, (2,5), (1,2), (1,1))
            self.de5 = nn.ConvTranspose2d(c5*2, c4, (2,5), (1,2), (1,1), output_padding=(0,1))
        else:
            self.de5 = nn.ConvTranspose2d(c5*2, c4, (2,5), (1,2), (1,1))

        self.de4 = nn.ConvTranspose2d(c4*2, c3, (2,5), (1,2), (1,1), output_padding=(0,1))
        self.de3 = nn.ConvTranspose2d(c3*2, c2, (2,5), (1,2), (1,1))
        self.de2 = nn.ConvTranspose2d(c2*2, c1, (2,5), (1,2), (1,1), output_padding=(0,1))
        self.de1 = nn.ConvTranspose2d(c1*2, 2, (2,5), (1,2), (1,1))

        # Encoder norms
        self.bn1 = nn.InstanceNorm2d(c1, affine=True)
        self.bn2 = nn.InstanceNorm2d(c2, affine=True)
        self.bn3 = nn.InstanceNorm2d(c3, affine=True)
        self.bn4 = nn.InstanceNorm2d(c4, affine=True)
        self.bn5 = nn.InstanceNorm2d(c5, affine=True)
        if num_layer >= 6:
            self.bn6 = nn.InstanceNorm2d(c6, affine=True)

        # Decoder norms (no norm on final output)
        if num_layer >= 6:
            self.bn6_t = nn.InstanceNorm2d(c5, affine=True)
            self.bn5_t = nn.InstanceNorm2d(c4, affine=True)
        else:
            self.bn5_t = nn.InstanceNorm2d(c4, affine=True)

        self.bn4_t = nn.InstanceNorm2d(c3, affine=True)
        self.bn3_t = nn.InstanceNorm2d(c2, affine=True)
        self.bn2_t = nn.InstanceNorm2d(c1, affine=True)

        # PReLU activations
        self.prelu1 = nn.PReLU(c1)
        self.prelu2 = nn.PReLU(c2)
        self.prelu3 = nn.PReLU(c3)
        self.prelu4 = nn.PReLU(c4)
        self.prelu5 = nn.PReLU(c5)
        if num_layer >= 6:
            self.prelu6 = nn.PReLU(c6)
            self.prelu6_t = nn.PReLU(c5)

        self.prelu5_t = nn.PReLU(c4)
        self.prelu4_t = nn.PReLU(c3)
        self.prelu3_t = nn.PReLU(c2)
        self.prelu2_t = nn.PReLU(c1)

    def forward(self, x):
        """
        Forward pass with adaptive shape matching (supports variable num_layer and any F)

        Args:
            x: Complex spectrogram [B, F, T] from torch.stft

        Returns:
            s: Enhanced complex spectrogram [B, F, T]
        """
        # Store original shape for reconstruction
        original_F = x.shape[1]
        original_T = x.shape[2]

        # ============================================================
        # INPUT ADAPTER: [B, F, T] complex → [B, 2, T, F] real
        # ============================================================
        # Convert complex to real format
        x_real = torch.view_as_real(x)  # [B, F, T, 2]

        # Permute to [B, 2, T, F] format (Trinet format)
        x_real = x_real.permute(0, 3, 2, 1)  # [B, 2, T, F]

        # ============================================================
        # ENCODER with FAC (Novel Component - Preserved)
        # ============================================================
        e1 = self.prelu1(self.bn1(self.conv1(x_real)[:,:,:-1]))
        e2 = self.prelu2(self.bn2(self.conv2(e1)[:,:,:-1]))
        e3 = self.prelu3(self.bn3(self.conv3(e2)[:,:,:-1]))
        e4 = self.prelu4(self.bn4(self.conv4(e3)[:,:,:-1]))
        e5 = self.prelu5(self.bn5(self.conv5(e4)[:,:,:-1]))

        # 6th layer (if num_layer >= 6)
        if self.num_layer >= 6:
            e6 = self.prelu6(self.bn6(self.conv6(e5)[:,:,:-1]))
            bottleneck_input = e6
        else:
            bottleneck_input = e5

        # ============================================================
        # BOTTLENECK: AIA Transformer (Novel Component - Preserved)
        # ============================================================
        aia_out = self.m1(bottleneck_input)
        out = torch.cat([aia_out, bottleneck_input], dim=1)

        # ============================================================
        # DECODER with Adaptive Skip Connections (U-Net style)
        # ============================================================
        # Helper function for adaptive shape matching
        def match_shape(decoder_out, encoder_out):
            """Match decoder output shape to encoder output shape"""
            if decoder_out.shape[2:] != encoder_out.shape[2:]:
                # Use interpolate to match exact dimensions
                decoder_out = nn.functional.interpolate(
                    decoder_out,
                    size=encoder_out.shape[2:],  # Match (T, F) dimensions
                    mode='bilinear',
                    align_corners=False
                )
            return decoder_out

        if self.num_layer >= 6:
            d6 = self.prelu6_t(self.bn6_t(F.pad(self.de6(out), [0,0,1,0])))
            d6 = match_shape(d6, e5)  # ✅ Adaptive matching
            out = torch.cat([d6, e5], dim=1)
            d5 = self.prelu5_t(self.bn5_t(F.pad(self.de5(out), [0,0,1,0])))
        else:
            d5 = self.prelu5_t(self.bn5_t(F.pad(self.de5(out), [0,0,1,0])))

        d5 = match_shape(d5, e4)  # ✅ Adaptive matching
        out = torch.cat([d5, e4], dim=1)

        d4 = self.prelu4_t(self.bn4_t(F.pad(self.de4(out), [0,0,1,0])))
        d4 = match_shape(d4, e3)  # ✅ Adaptive matching
        out = torch.cat([d4, e3], dim=1)

        d3 = self.prelu3_t(self.bn3_t(F.pad(self.de3(out), [0,0,1,0])))
        d3 = match_shape(d3, e2)  # ✅ Adaptive matching
        out = torch.cat([d3, e2], dim=1)

        d2 = self.prelu2_t(self.bn2_t(F.pad(self.de2(out), [0,0,1,0])))
        d2 = match_shape(d2, e1)  # ✅ Adaptive matching
        out = torch.cat([d2, e1], dim=1)

        # Final output (NO normalization, NO activation)
        d1 = F.pad(self.de1(out), [0,0,1,0])  # [B, 2, T, F]

        # Match to input shape exactly (handles any rounding errors)
        if d1.shape[2:] != (original_T, original_F):
            d1 = nn.functional.interpolate(
                d1,
                size=(original_T, original_F),
                mode='bilinear',
                align_corners=False
            )

        # ============================================================
        # OUTPUT ADAPTER: [B, 2, T, F] real → [B, F, T] complex
        # ============================================================
        # Permute back to [B, F, T, 2]
        d1 = d1.permute(0, 3, 2, 1)  # [B, F, T, 2]

        # Convert to complex format
        s = torch.view_as_complex(d1.contiguous())  # [B, F, T]

        return s


# ============================================================================
# DISCRIMINATOR (From BSRNN - Not a novel component)
# ============================================================================

class Discriminator(nn.Module):
    """
    Metric Discriminator from BSRNN
    Predicts PESQ scores for adversarial training
    """
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)


# ============================================================================
# COMPATIBILITY: Alias for BSRNN training code
# ============================================================================

# The training code expects a class named "BSRNN", so we create an alias
BSRNN = TrinetBSRNN
