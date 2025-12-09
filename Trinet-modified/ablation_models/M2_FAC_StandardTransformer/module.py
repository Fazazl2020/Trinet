"""
M2: ABLATION - FAC + Standard Transformer
==========================================

This model tests the contribution of FAC (Frequency-Adaptive Convolution) alone.
- Uses FAC in encoder (novel component)
- Uses standard multi-head self-attention transformer (no multi-resolution)
- Same U-Net structure as proposed model

Purpose: Isolate the contribution of FAC without multi-resolution attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import LearnableSigmoid


# ============================================================================
# NOVEL COMPONENT: FAC (Frequency-Adaptive Convolution)
# ============================================================================

class AdaptiveFrequencyBandPositionalEncoding(nn.Module):
    """
    NOVEL COMPONENT - Learnable positional encoding for different frequency bands.
    """
    def __init__(self, F=257, sample_rate=16000, d_pe=16):
        super().__init__()
        self.sample_rate = sample_rate
        self.F_init = F
        self.d_pe = d_pe

        self.low_freq_range = (0, 300)
        self.mid_freq_range = (300, 3400)
        self.high_freq_range = (3400, sample_rate / 2)

        nyquist = sample_rate / 2
        self.low_bins_init = int(F * (self.low_freq_range[1] / nyquist))
        self.mid_bins_init = int(F * (self.mid_freq_range[1] / nyquist)) - self.low_bins_init
        self.high_bins_init = F - (self.low_bins_init + self.mid_bins_init)

        self.pe_low = self._init_multiscale_pe(self.low_bins_init, offset=0)
        self.pe_mid = self._init_multiscale_pe(self.mid_bins_init, offset=self.low_bins_init)
        self.pe_high = self._init_multiscale_pe(self.high_bins_init, offset=self.low_bins_init + self.mid_bins_init)

        self.band_weights = nn.Parameter(torch.tensor([0.5, 1.0, 0.3]))

    def _init_multiscale_pe(self, num_bins, offset):
        if num_bins <= 0:
            return None

        position = torch.arange(num_bins).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_pe, 2).float() * (-math.log(10000.0) / self.d_pe)
        )

        pe = torch.zeros(num_bins, self.d_pe)
        pe[:, 0::2] = torch.sin((position + offset) * div_term)
        pe[:, 1::2] = torch.cos((position + offset) * div_term)

        pe_combined = pe.mean(dim=1)
        pe_combined = (pe_combined - pe_combined.min()) / (pe_combined.max() - pe_combined.min() + 1e-8)

        return nn.Parameter(pe_combined, requires_grad=True)

    def forward(self, X):
        batch_size, C, T, freq_bins = X.size()
        device = X.device
        nyquist = self.sample_rate / 2

        current_low = int(freq_bins * (self.low_freq_range[1] / nyquist))
        current_mid = int(freq_bins * (self.mid_freq_range[1] / nyquist)) - current_low
        current_high = freq_bins - (current_low + current_mid)

        pe_adaptive = torch.zeros(batch_size, C, T, freq_bins, device=device)
        band_weights = F.softmax(self.band_weights, dim=0) * 3.0

        if self.pe_low is not None and current_low > 0:
            pe_low = F.interpolate(
                self.pe_low.unsqueeze(0).unsqueeze(0),
                size=current_low, mode='linear', align_corners=False
            )
            pe_low = pe_low.unsqueeze(2) * band_weights[0]
            pe_adaptive[:, :, :, :current_low] = pe_low.expand(batch_size, C, T, current_low)

        if self.pe_mid is not None and current_mid > 0:
            pe_mid = F.interpolate(
                self.pe_mid.unsqueeze(0).unsqueeze(0),
                size=current_mid, mode='linear', align_corners=False
            )
            pe_mid = pe_mid.unsqueeze(2) * band_weights[1]
            pe_adaptive[:, :, :, current_low:current_low + current_mid] = pe_mid.expand(batch_size, C, T, current_mid)

        if self.pe_high is not None and current_high > 0:
            pe_high = F.interpolate(
                self.pe_high.unsqueeze(0).unsqueeze(0),
                size=current_high, mode='linear', align_corners=False
            )
            pe_high = pe_high.unsqueeze(2) * band_weights[2]
            pe_adaptive[:, :, :, current_low + current_mid:] = pe_high.expand(batch_size, C, T, current_high)

        return pe_adaptive


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


class GatedPositionalEncoding(nn.Module):
    """NOVEL COMPONENT - Gated PE with adaptive scaling"""
    def __init__(self, in_channels, F=257, sample_rate=16000):
        super().__init__()
        self.positional_encoding = AdaptiveFrequencyBandPositionalEncoding(F=F, sample_rate=sample_rate)
        self.dw_attention = DepthwiseFrequencyAttention(in_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.scale_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, X):
        P_freq = self.positional_encoding(X)
        attn = self.dw_attention(X)
        gate = self.gate(X)

        scale = self.scale_net(X)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        P_freq_scaled = P_freq * scale

        return X + gate * attn * P_freq_scaled


class FACLayer(nn.Module):
    """NOVEL COMPONENT - Frequency-Adaptive Convolution Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, F=257, sample_rate=16000):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gated_pe = GatedPositionalEncoding(in_channels, F=F, sample_rate=sample_rate)

    def forward(self, X):
        X = self.gated_pe(X)
        return self.conv(X)


# ============================================================================
# STANDARD TRANSFORMER (from M1)
# ============================================================================

class StandardTransformer(nn.Module):
    """Standard Transformer with multi-head self-attention"""
    def __init__(self, input_size, output_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = input_size // 2
        self.num_heads = num_heads

        self.input_proj = nn.Sequential(
            nn.Conv2d(input_size, self.hidden_size, kernel_size=1),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(self.hidden_size)

        self.output_proj = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(self.hidden_size, output_size, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, T, F = x.shape

        x = self.input_proj(x)
        _, C_hidden, _, _ = x.shape

        x_seq = x.permute(0, 2, 3, 1).contiguous().view(B, T * F, C_hidden)

        attn_out, _ = self.multihead_attn(x_seq, x_seq, x_seq)

        x_seq = self.norm(x_seq + attn_out)

        x_out = x_seq.view(B, T, F, C_hidden).permute(0, 3, 1, 2).contiguous()

        return self.output_proj(x_out)


# ============================================================================
# MAIN NETWORK: FAC + Standard Transformer
# ============================================================================

class TrinetBSRNN(nn.Module):
    """
    M2: FAC + Standard Transformer

    Tests FAC contribution alone:
    - Uses FAC in encoder (novel)
    - Uses standard transformer in bottleneck (baseline)
    """
    def __init__(self, num_channel=128, num_layer=6, F=257):
        super().__init__()
        self.num_channel = num_channel
        self.num_layer = num_layer
        self.F = F

        scale = num_channel / 128.0
        c1 = max(8, int(16 * scale))
        c2 = max(16, int(32 * scale))
        c3 = max(32, int(64 * scale))
        c4 = max(64, int(128 * scale))
        c5 = max(128, int(256 * scale))
        c6 = max(256, int(512 * scale)) if num_layer >= 6 else c5
        bottleneck_ch = c6 if num_layer >= 6 else c5

        self.channels = [2, c1, c2, c3, c4, c5]
        if num_layer >= 6:
            self.channels.append(c6)

        # ENCODER with FAC (NOVEL)
        self.conv1 = FACLayer(2, c1, (2,5), (1,2), (1,1), F)
        self.conv2 = FACLayer(c1, c2, (2,5), (1,2), (1,1), F)
        self.conv3 = FACLayer(c2, c3, (2,5), (1,2), (1,1), F)
        self.conv4 = FACLayer(c3, c4, (2,5), (1,2), (1,1), F)
        self.conv5 = FACLayer(c4, c5, (2,5), (1,2), (1,1), F)

        if num_layer >= 6:
            self.conv6 = FACLayer(c5, c6, (2,5), (1,2), (1,1), F)

        # BOTTLENECK: Standard Transformer (BASELINE)
        self.m1 = StandardTransformer(bottleneck_ch, bottleneck_ch)

        # DECODER
        if num_layer >= 6:
            self.de6 = nn.ConvTranspose2d(c6*2, c5, (2,5), (1,2), (1,1))
            self.de5 = nn.ConvTranspose2d(c5*2, c4, (2,5), (1,2), (1,1), output_padding=(0,1))
        else:
            self.de5 = nn.ConvTranspose2d(c5*2, c4, (2,5), (1,2), (1,1))

        self.de4 = nn.ConvTranspose2d(c4*2, c3, (2,5), (1,2), (1,1), output_padding=(0,1))
        self.de3 = nn.ConvTranspose2d(c3*2, c2, (2,5), (1,2), (1,1))
        self.de2 = nn.ConvTranspose2d(c2*2, c1, (2,5), (1,2), (1,1), output_padding=(0,1))
        self.de1 = nn.ConvTranspose2d(c1*2, 2, (2,5), (1,2), (1,1))

        # Norms and activations (same as original)
        self.bn1 = nn.InstanceNorm2d(c1, affine=True)
        self.bn2 = nn.InstanceNorm2d(c2, affine=True)
        self.bn3 = nn.InstanceNorm2d(c3, affine=True)
        self.bn4 = nn.InstanceNorm2d(c4, affine=True)
        self.bn5 = nn.InstanceNorm2d(c5, affine=True)
        if num_layer >= 6:
            self.bn6 = nn.InstanceNorm2d(c6, affine=True)

        if num_layer >= 6:
            self.bn6_t = nn.InstanceNorm2d(c5, affine=True)
            self.bn5_t = nn.InstanceNorm2d(c4, affine=True)
        else:
            self.bn5_t = nn.InstanceNorm2d(c4, affine=True)

        self.bn4_t = nn.InstanceNorm2d(c3, affine=True)
        self.bn3_t = nn.InstanceNorm2d(c2, affine=True)
        self.bn2_t = nn.InstanceNorm2d(c1, affine=True)

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
        original_F = x.shape[1]
        original_T = x.shape[2]

        x_real = torch.view_as_real(x)
        x_real = x_real.permute(0, 3, 2, 1)

        # ENCODER with FAC
        e1 = self.prelu1(self.bn1(self.conv1(x_real)[:,:,:-1]))
        e2 = self.prelu2(self.bn2(self.conv2(e1)[:,:,:-1]))
        e3 = self.prelu3(self.bn3(self.conv3(e2)[:,:,:-1]))
        e4 = self.prelu4(self.bn4(self.conv4(e3)[:,:,:-1]))
        e5 = self.prelu5(self.bn5(self.conv5(e4)[:,:,:-1]))

        if self.num_layer >= 6:
            e6 = self.prelu6(self.bn6(self.conv6(e5)[:,:,:-1]))
            bottleneck_input = e6
        else:
            bottleneck_input = e5

        # BOTTLENECK: Standard Transformer
        transformer_out = self.m1(bottleneck_input)
        out = torch.cat([transformer_out, bottleneck_input], dim=1)

        # DECODER
        def match_shape(decoder_out, encoder_out):
            if decoder_out.shape[2:] != encoder_out.shape[2:]:
                decoder_out = nn.functional.interpolate(
                    decoder_out,
                    size=encoder_out.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return decoder_out

        if self.num_layer >= 6:
            d6 = self.prelu6_t(self.bn6_t(F.pad(self.de6(out), [0,0,1,0])))
            d6 = match_shape(d6, e5)
            out = torch.cat([d6, e5], dim=1)
            d5 = self.prelu5_t(self.bn5_t(F.pad(self.de5(out), [0,0,1,0])))
        else:
            d5 = self.prelu5_t(self.bn5_t(F.pad(self.de5(out), [0,0,1,0])))

        d5 = match_shape(d5, e4)
        out = torch.cat([d5, e4], dim=1)

        d4 = self.prelu4_t(self.bn4_t(F.pad(self.de4(out), [0,0,1,0])))
        d4 = match_shape(d4, e3)
        out = torch.cat([d4, e3], dim=1)

        d3 = self.prelu3_t(self.bn3_t(F.pad(self.de3(out), [0,0,1,0])))
        d3 = match_shape(d3, e2)
        out = torch.cat([d3, e2], dim=1)

        d2 = self.prelu2_t(self.bn2_t(F.pad(self.de2(out), [0,0,1,0])))
        d2 = match_shape(d2, e1)
        out = torch.cat([d2, e1], dim=1)

        d1 = F.pad(self.de1(out), [0,0,1,0])

        if d1.shape[2:] != (original_T, original_F):
            d1 = nn.functional.interpolate(
                d1,
                size=(original_T, original_F),
                mode='bilinear',
                align_corners=False
            )

        d1 = d1.permute(0, 3, 2, 1)
        s = torch.view_as_complex(d1.contiguous())

        return s


# ============================================================================
# DISCRIMINATOR (Same as original)
# ============================================================================

class Discriminator(nn.Module):
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


BSRNN = TrinetBSRNN
