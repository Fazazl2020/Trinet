"""
M1: ABLATION BASELINE - Standard Conv2D + Standard Transformer
================================================================

This is the BASELINE model for ablation study.
- Uses standard Conv2D (no frequency-adaptive components)
- Uses standard multi-head self-attention transformer (no multi-resolution)
- Same U-Net structure as proposed model

Purpose: Establish baseline performance without novel components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import LearnableSigmoid


# ============================================================================
# STANDARD COMPONENTS (Baseline)
# ============================================================================

class StandardConvLayer(nn.Module):
    """Standard Conv2D layer without frequency-adaptive components"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, F=257, sample_rate=16000):
        super().__init__()
        # Simple Conv2D - NO positional encoding, NO gating, NO frequency attention
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, X):
        # Direct convolution without any frequency-adaptive processing
        return self.conv(X)


class StandardTransformer(nn.Module):
    """Standard Transformer with multi-head self-attention"""
    def __init__(self, input_size, output_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = input_size // 2
        self.num_heads = num_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_size, self.hidden_size, kernel_size=1),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Standard multi-head self-attention
        # Process as sequence: flatten spatial dimensions
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Normalization
        self.norm = nn.LayerNorm(self.hidden_size)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(self.hidden_size, output_size, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, T, F = x.shape

        # Input projection
        x = self.input_proj(x)  # [B, hidden_size, T, F]
        _, C_hidden, _, _ = x.shape

        # Reshape for attention: [B, T*F, C_hidden]
        x_seq = x.permute(0, 2, 3, 1).contiguous()  # [B, T, F, C_hidden]
        x_seq = x_seq.view(B, T * F, C_hidden)  # [B, T*F, C_hidden]

        # Standard multi-head self-attention
        attn_out, _ = self.multihead_attn(x_seq, x_seq, x_seq)  # [B, T*F, C_hidden]

        # Residual connection + normalization
        x_seq = self.norm(x_seq + attn_out)

        # Reshape back to 2D: [B, C_hidden, T, F]
        x_out = x_seq.view(B, T, F, C_hidden).permute(0, 3, 1, 2).contiguous()

        # Output projection
        return self.output_proj(x_out)


# ============================================================================
# MAIN NETWORK: Baseline U-Net
# ============================================================================

class TrinetBSRNN(nn.Module):
    """
    M1: ABLATION BASELINE

    Standard U-Net with Conv2D and standard Transformer
    - NO frequency-adaptive convolution
    - NO multi-resolution attention
    - NO cosine attention
    - NO adaptive gating

    Same structure as proposed model but with standard components.
    """
    def __init__(self, num_channel=128, num_layer=6, F=257):
        super().__init__()
        self.num_channel = num_channel
        self.num_layer = num_layer
        self.F = F

        # Calculate channel progression
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

        # ============================================================
        # ENCODER with Standard Conv2D (NO FAC)
        # ============================================================
        self.conv1 = StandardConvLayer(2, c1, (2,5), (1,2), (1,1), F)
        self.conv2 = StandardConvLayer(c1, c2, (2,5), (1,2), (1,1), F)
        self.conv3 = StandardConvLayer(c2, c3, (2,5), (1,2), (1,1), F)
        self.conv4 = StandardConvLayer(c3, c4, (2,5), (1,2), (1,1), F)
        self.conv5 = StandardConvLayer(c4, c5, (2,5), (1,2), (1,1), F)

        if num_layer >= 6:
            self.conv6 = StandardConvLayer(c5, c6, (2,5), (1,2), (1,1), F)

        # ============================================================
        # BOTTLENECK: Standard Transformer (NO AIA)
        # ============================================================
        self.m1 = StandardTransformer(bottleneck_ch, bottleneck_ch)

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

        # Decoder norms
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
        """Forward pass - identical structure to proposed model"""
        # Store original shape
        original_F = x.shape[1]
        original_T = x.shape[2]

        # INPUT ADAPTER: [B, F, T] complex → [B, 2, T, F] real
        x_real = torch.view_as_real(x)  # [B, F, T, 2]
        x_real = x_real.permute(0, 3, 2, 1)  # [B, 2, T, F]

        # ENCODER (Standard Conv2D)
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

        # BOTTLENECK (Standard Transformer)
        transformer_out = self.m1(bottleneck_input)
        out = torch.cat([transformer_out, bottleneck_input], dim=1)

        # DECODER with adaptive skip connections
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

        # Match to input shape
        if d1.shape[2:] != (original_T, original_F):
            d1 = nn.functional.interpolate(
                d1,
                size=(original_T, original_F),
                mode='bilinear',
                align_corners=False
            )

        # OUTPUT ADAPTER: [B, 2, T, F] real → [B, F, T] complex
        d1 = d1.permute(0, 3, 2, 1)  # [B, F, T, 2]
        s = torch.view_as_complex(d1.contiguous())  # [B, F, T]

        return s


# ============================================================================
# DISCRIMINATOR (Same as original)
# ============================================================================

class Discriminator(nn.Module):
    """Metric Discriminator - unchanged from original"""
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
# COMPATIBILITY
# ============================================================================

BSRNN = TrinetBSRNN
