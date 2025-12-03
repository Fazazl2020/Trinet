"""
Dimension Analysis Script
=========================
Traces dimensions through TrinetBSRNN encoder/decoder
to identify skip connection mismatches.
"""

import math

def conv2d_output_size(input_size, kernel, stride, padding):
    """Calculate Conv2d output size"""
    return math.floor((input_size + 2*padding - kernel) / stride) + 1

def convtranspose2d_output_size(input_size, kernel, stride, padding, output_padding):
    """Calculate ConvTranspose2d output size"""
    return (input_size - 1) * stride - 2*padding + kernel + output_padding

# Configuration
num_channel = 64
num_layer = 5
F_init = 257  # n_fft=512 → F = 512//2 + 1 = 257
T_init = 250  # Example time frames

# Channel progression
scale = num_channel / 128.0
c1 = max(8, int(16 * scale))   # 8
c2 = max(16, int(32 * scale))  # 16
c3 = max(32, int(64 * scale))  # 32
c4 = max(64, int(128 * scale)) # 64
c5 = max(128, int(256 * scale)) # 128

print("="*80)
print("DIMENSION ANALYSIS - TrinetBSRNN")
print("="*80)
print(f"Configuration: num_channel={num_channel}, num_layer={num_layer}")
print(f"Input: [B, F={F_init}, T={T_init}] complex")
print(f"Channels: c1={c1}, c2={c2}, c3={c3}, c4={c4}, c5={c5}")
print("="*80)

# Convert to real format
print("\n1. INPUT ADAPTER")
print(f"   Complex [B, {F_init}, {T_init}] → Real [B, {F_init}, {T_init}, 2]")
print(f"   Permute → [B, 2, {T_init}, {F_init}]")

T, F = T_init, F_init

# Encoder dimensions
print("\n2. ENCODER (kernel=(2,5), stride=(1,2), padding=(1,1))")
print("-"*80)

encoder_shapes = []

# Layer 1
T_out = conv2d_output_size(T, 2, 1, 1)
F_out = conv2d_output_size(F, 5, 2, 1)
print(f"   conv1: [B, 2, {T}, {F}] → [B, {c1}, {T_out}, {F_out}]")
T_out -= 1  # [:,:,:-1]
print(f"   After [:,:,:-1]: e1 = [B, {c1}, {T_out}, {F_out}]")
encoder_shapes.append((T_out, F_out))
T, F = T_out, F_out

# Layer 2
T_out = conv2d_output_size(T, 2, 1, 1)
F_out = conv2d_output_size(F, 5, 2, 1)
print(f"   conv2: [B, {c1}, {T}, {F}] → [B, {c2}, {T_out}, {F_out}]")
T_out -= 1
print(f"   After [:,:,:-1]: e2 = [B, {c2}, {T_out}, {F_out}]")
encoder_shapes.append((T_out, F_out))
T, F = T_out, F_out

# Layer 3
T_out = conv2d_output_size(T, 2, 1, 1)
F_out = conv2d_output_size(F, 5, 2, 1)
print(f"   conv3: [B, {c2}, {T}, {F}] → [B, {c3}, {T_out}, {F_out}]")
T_out -= 1
print(f"   After [:,:,:-1]: e3 = [B, {c3}, {T_out}, {F_out}]")
encoder_shapes.append((T_out, F_out))
T, F = T_out, F_out

# Layer 4
T_out = conv2d_output_size(T, 2, 1, 1)
F_out = conv2d_output_size(F, 5, 2, 1)
print(f"   conv4: [B, {c3}, {T}, {F}] → [B, {c4}, {T_out}, {F_out}]")
T_out -= 1
print(f"   After [:,:,:-1]: e4 = [B, {c4}, {T_out}, {F_out}]")
encoder_shapes.append((T_out, F_out))
T, F = T_out, F_out

# Layer 5
T_out = conv2d_output_size(T, 2, 1, 1)
F_out = conv2d_output_size(F, 5, 2, 1)
print(f"   conv5: [B, {c4}, {T}, {F}] → [B, {c5}, {T_out}, {F_out}]")
T_out -= 1
print(f"   After [:,:,:-1]: e5 = [B, {c5}, {T_out}, {F_out}]")
encoder_shapes.append((T_out, F_out))
T, F = T_out, F_out

print(f"\n   Bottleneck input: [B, {c5}, {T}, {F}]")
print(f"   After AIA_Transformer + concat: [B, {c5*2}, {T}, {F}]")

# Decoder dimensions
print("\n3. DECODER (kernel=(2,5), stride=(1,2), padding=(1,1))")
print("-"*80)

# Current output_padding from code
output_paddings = [(0,0), (0,1), (0,0), (0,1), (0,0)]

# Layer de5
T_out = convtranspose2d_output_size(T, 2, 1, 1, output_paddings[0][0])
F_out = convtranspose2d_output_size(F, 5, 2, 1, output_paddings[0][1])
print(f"   de5: [B, {c5*2}, {T}, {F}] → [B, {c4}, {T_out}, {F_out}]")
T_out += 1  # F.pad([0,0,1,0])
print(f"   After pad: d5 = [B, {c4}, {T_out}, {F_out}]")
e4_T, e4_F = encoder_shapes[3]
print(f"   Skip from e4: [B, {c4}, {e4_T}, {e4_F}]")
if T_out == e4_T and F_out == e4_F:
    print(f"   ✅ MATCH: Can concatenate")
else:
    print(f"   ❌ MISMATCH: T:{T_out}≠{e4_T}, F:{F_out}≠{e4_F}")
T, F = T_out, F_out

# Layer de4
print(f"   After concat: [B, {c4*2}, {T}, {F}]")
T_out = convtranspose2d_output_size(T, 2, 1, 1, output_paddings[1][0])
F_out = convtranspose2d_output_size(F, 5, 2, 1, output_paddings[1][1])
print(f"   de4: [B, {c4*2}, {T}, {F}] → [B, {c3}, {T_out}, {F_out}]")
T_out += 1
print(f"   After pad: d4 = [B, {c3}, {T_out}, {F_out}]")
e3_T, e3_F = encoder_shapes[2]
print(f"   Skip from e3: [B, {c3}, {e3_T}, {e3_F}]")
if T_out == e3_T and F_out == e3_F:
    print(f"   ✅ MATCH: Can concatenate")
else:
    print(f"   ❌ MISMATCH: T:{T_out}≠{e3_T}, F:{F_out}≠{e3_F}")
T, F = T_out, F_out

# Layer de3
print(f"   After concat: [B, {c3*2}, {T}, {F}]")
T_out = convtranspose2d_output_size(T, 2, 1, 1, output_paddings[2][0])
F_out = convtranspose2d_output_size(F, 5, 2, 1, output_paddings[2][1])
print(f"   de3: [B, {c3*2}, {T}, {F}] → [B, {c2}, {T_out}, {F_out}]")
T_out += 1
print(f"   After pad: d3 = [B, {c2}, {T_out}, {F_out}]")
e2_T, e2_F = encoder_shapes[1]
print(f"   Skip from e2: [B, {c2}, {e2_T}, {e2_F}]")
if T_out == e2_T and F_out == e2_F:
    print(f"   ✅ MATCH: Can concatenate")
else:
    print(f"   ❌ MISMATCH: T:{T_out}≠{e2_T}, F:{F_out}≠{e2_F}")
T, F = T_out, F_out

# Layer de2
print(f"   After concat: [B, {c2*2}, {T}, {F}]")
T_out = convtranspose2d_output_size(T, 2, 1, 1, output_paddings[3][0])
F_out = convtranspose2d_output_size(F, 5, 2, 1, output_paddings[3][1])
print(f"   de2: [B, {c2*2}, {T}, {F}] → [B, {c1}, {T_out}, {F_out}]")
T_out += 1
print(f"   After pad: d2 = [B, {c1}, {T_out}, {F_out}]")
e1_T, e1_F = encoder_shapes[0]
print(f"   Skip from e1: [B, {c1}, {e1_T}, {e1_F}]")
if T_out == e1_T and F_out == e1_F:
    print(f"   ✅ MATCH: Can concatenate")
else:
    print(f"   ❌ MISMATCH: T:{T_out}≠{e1_T}, F:{F_out}≠{e1_F}")
T, F = T_out, F_out

# Layer de1
print(f"   After concat: [B, {c1*2}, {T}, {F}]")
T_out = convtranspose2d_output_size(T, 2, 1, 1, output_paddings[4][0])
F_out = convtranspose2d_output_size(F, 5, 2, 1, output_paddings[4][1])
print(f"   de1: [B, {c1*2}, {T}, {F}] → [B, 2, {T_out}, {F_out}]")
T_out += 1
print(f"   After pad: d1 = [B, 2, {T_out}, {F_out}]")
print(f"   Should reconstruct to: [B, 2, {T_init}, {F_init}]")
if T_out == T_init and F_out == F_init:
    print(f"   ✅ MATCH: Correct reconstruction")
else:
    print(f"   ❌ MISMATCH: T:{T_out}≠{T_init}, F:{F_out}≠{F_init}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Encoder skip connection shapes:")
for i, (t, f) in enumerate(encoder_shapes, 1):
    print(f"  e{i}: [*, {t}, {f}]")

print("\nRequired decoder output_padding to match:")
print("Need to calculate dynamically based on encoder shapes!")
print("="*80)
