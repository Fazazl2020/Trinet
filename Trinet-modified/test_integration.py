"""
Comprehensive Test Suite for TrinetBSRNN Integration
=====================================================

This script verifies that the Trinet network is correctly integrated with the BSRNN pipeline.

Tests:
1. Shape compatibility (input/output formats)
2. Novel components preservation (FAC, AIA_Transformer)
3. Forward pass (no errors)
4. Gradient flow (backpropagation works)
5. Integration with BSRNN training pipeline
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module import TrinetBSRNN, BSRNN, Discriminator
from module import FACLayer, AIA_Transformer, AdaptiveFrequencyBandPositionalEncoding


def test_1_shape_compatibility():
    """Test 1: Verify input/output shape compatibility with BSRNN pipeline"""
    print("\n" + "="*70)
    print("TEST 1: Shape Compatibility")
    print("="*70)

    batch_size = 2
    n_fft = 512
    hop = 128
    cut_len = int(16000 * 2)  # 2 seconds

    F = n_fft // 2 + 1  # 257 frequency bins
    T = cut_len // hop + 1  # Number of time frames

    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  n_fft: {n_fft}")
    print(f"  hop: {hop}")
    print(f"  cut_len: {cut_len} samples")
    print(f"  F (freq bins): {F}")
    print(f"  T (time frames): {T}")

    # Create model
    model = TrinetBSRNN(F=F)
    model.eval()

    # Create dummy input (BSRNN format: complex spectrogram)
    # This simulates the output of torch.stft
    dummy_input = torch.randn(batch_size, F, T, dtype=torch.cfloat)

    print(f"\nInput shape: {dummy_input.shape} (complex)")
    print(f"Input dtype: {dummy_input.dtype}")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nOutput shape: {output.shape} (complex)")
    print(f"Output dtype: {output.dtype}")

    # Verify shapes match
    assert output.shape == dummy_input.shape, f"Shape mismatch! Input: {dummy_input.shape}, Output: {output.shape}"
    assert output.dtype == torch.cfloat, f"Output should be complex, got {output.dtype}"

    print("\n✅ TEST 1 PASSED: Shapes are compatible with BSRNN pipeline")
    return True


def test_2_novel_components():
    """Test 2: Verify novel components (FAC, AIA_Transformer) are present and working"""
    print("\n" + "="*70)
    print("TEST 2: Novel Components Preservation")
    print("="*70)

    model = TrinetBSRNN(F=257)

    # Check FAC components
    print("\n1. Checking FAC (Frequency-Adaptive Convolution)...")
    has_fac = False
    for name, module in model.named_modules():
        if isinstance(module, FACLayer):
            has_fac = True
            print(f"   ✓ Found FACLayer: {name}")

            # Check sub-components
            assert hasattr(module, 'gated_pe'), "FACLayer missing gated_pe"
            assert hasattr(module.gated_pe, 'positional_encoding'), "Missing positional_encoding"
            assert isinstance(module.gated_pe.positional_encoding, AdaptiveFrequencyBandPositionalEncoding), \
                "Wrong PE type"

    assert has_fac, "No FAC layers found!"
    print("   ✅ FAC components present and correct")

    # Check AIA_Transformer
    print("\n2. Checking AIA_Transformer (Bottleneck Transformer)...")
    has_aia = False
    for name, module in model.named_modules():
        if isinstance(module, AIA_Transformer):
            has_aia = True
            print(f"   ✓ Found AIA_Transformer: {name}")

            # Check sub-components
            assert hasattr(module, 'row_trans'), "Missing row_trans"
            assert hasattr(module, 'col_trans'), "Missing col_trans"
            assert hasattr(module, 'pe_scale'), "Missing pe_scale"

    assert has_aia, "No AIA_Transformer found!"
    print("   ✅ AIA_Transformer present and correct")

    # Check learnable parameters
    print("\n3. Checking novel learnable parameters...")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total trainable parameters: {param_count:,}")

    # Check for specific novel parameters
    novel_params = []
    for name, param in model.named_parameters():
        if 'band_weights' in name or 'cos_temperature' in name or 'pe_scale' in name:
            novel_params.append(name)
            print(f"   ✓ Found novel parameter: {name} (shape: {param.shape})")

    assert len(novel_params) > 0, "No novel parameters found!"
    print(f"   ✅ Found {len(novel_params)} novel parameters")

    print("\n✅ TEST 2 PASSED: Novel components preserved")
    return True


def test_3_forward_pass():
    """Test 3: Verify forward pass works without errors"""
    print("\n" + "="*70)
    print("TEST 3: Forward Pass")
    print("="*70)

    batch_size = 4
    F = 257
    T = 250

    model = TrinetBSRNN(F=F)
    model.eval()

    # Create dummy input
    x = torch.randn(batch_size, F, T, dtype=torch.cfloat)

    print(f"Running forward pass...")
    print(f"  Input: {x.shape}")

    try:
        with torch.no_grad():
            output = model(x)
        print(f"  Output: {output.shape}")
        print("  ✅ Forward pass successful")

        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN!"
        assert not torch.isinf(output).any(), "Output contains Inf!"
        print("  ✅ No NaN or Inf values")

    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        raise

    print("\n✅ TEST 3 PASSED: Forward pass works correctly")
    return True


def test_4_gradient_flow():
    """Test 4: Verify gradients flow properly through the network"""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow")
    print("="*70)

    batch_size = 2
    F = 257
    T = 250

    model = TrinetBSRNN(F=F)
    model.train()

    # Create dummy input and target
    x = torch.randn(batch_size, F, T, dtype=torch.cfloat)
    target = torch.randn(batch_size, F, T, dtype=torch.cfloat)

    print("Running backward pass...")

    # Forward pass
    output = model(x)

    # Compute simple loss (L1 on real and imaginary parts)
    loss_real = nn.L1Loss()(output.real, target.real)
    loss_imag = nn.L1Loss()(output.imag, target.imag)
    loss = loss_real + loss_imag

    print(f"  Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    params_with_grad = 0
    params_without_grad = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
                # Check for NaN in gradients
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            else:
                params_without_grad += 1
                print(f"  ⚠️  No gradient for: {name}")

    print(f"\n  Parameters with gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")

    assert params_with_grad > 0, "No parameters received gradients!"

    if params_without_grad > 0:
        print(f"  ⚠️  Warning: {params_without_grad} parameters have no gradients")
    else:
        print("  ✅ All parameters have gradients")

    print("\n✅ TEST 4 PASSED: Gradients flow correctly")
    return True


def test_5_bsrnn_training_compatibility():
    """Test 5: Verify compatibility with BSRNN training loop"""
    print("\n" + "="*70)
    print("TEST 5: BSRNN Training Pipeline Compatibility")
    print("="*70)

    # Parameters from BSRNN training
    n_fft = 512
    hop = 128
    batch_size = 2
    cut_len = int(16000 * 2)

    print("Simulating BSRNN training loop...")

    # Create models (using BSRNN alias)
    model = BSRNN(F=n_fft//2+1).cuda() if torch.cuda.is_available() else BSRNN(F=n_fft//2+1)
    discriminator = Discriminator(ndf=16)
    if torch.cuda.is_available():
        discriminator = discriminator.cuda()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Create dummy audio
    clean = torch.randn(batch_size, cut_len)
    noisy = torch.randn(batch_size, cut_len)

    if torch.cuda.is_available():
        clean = clean.cuda()
        noisy = noisy.cuda()

    print(f"  Input audio: {noisy.shape}")

    # Simulate BSRNN training step
    try:
        # STFT (same as BSRNN)
        window = torch.hann_window(n_fft)
        if torch.cuda.is_available():
            window = window.cuda()

        noisy_spec = torch.stft(noisy, n_fft, hop, window=window, onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, n_fft, hop, window=window, onesided=True, return_complex=True)

        print(f"  Spectrogram: {noisy_spec.shape}")

        # Forward pass through model
        est_spec = model(noisy_spec)
        print(f"  Estimated spec: {est_spec.shape}")

        # Compute magnitude for discriminator
        est_mag = (torch.abs(est_spec).unsqueeze(1) + 1e-10) ** 0.3
        clean_mag = (torch.abs(clean_spec).unsqueeze(1) + 1e-10) ** 0.3

        print(f"  Magnitude: {est_mag.shape}")

        # Discriminator forward pass
        disc_output = discriminator(clean_mag, est_mag)
        print(f"  Discriminator output: {disc_output.shape}")

        # Compute losses (same as BSRNN)
        mae_loss = nn.L1Loss()
        loss_mag = mae_loss(est_mag, clean_mag)
        loss_ri = mae_loss(est_spec, clean_spec)

        print(f"\n  Loss (magnitude): {loss_mag.item():.6f}")
        print(f"  Loss (RI): {loss_ri.item():.6f}")

        # ISTFT
        est_audio = torch.istft(est_spec, n_fft, hop, window=window, onesided=True)
        print(f"  Resynthesized audio: {est_audio.shape}")

        print("\n  ✅ All BSRNN pipeline steps successful")

    except Exception as e:
        print(f"\n  ❌ BSRNN pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n✅ TEST 5 PASSED: Fully compatible with BSRNN training pipeline")
    return True


def test_6_alias_verification():
    """Test 6: Verify BSRNN alias works correctly"""
    print("\n" + "="*70)
    print("TEST 6: BSRNN Alias Verification")
    print("="*70)

    # Check that BSRNN is an alias for TrinetBSRNN
    assert BSRNN is TrinetBSRNN, "BSRNN is not an alias for TrinetBSRNN!"
    print("  ✅ BSRNN is correctly aliased to TrinetBSRNN")

    # Create model using alias
    model1 = BSRNN()
    model2 = TrinetBSRNN()

    print(f"  BSRNN type: {type(model1)}")
    print(f"  TrinetBSRNN type: {type(model2)}")

    assert type(model1) == type(model2), "Types don't match!"
    print("  ✅ Both create the same type")

    print("\n✅ TEST 6 PASSED: Alias works correctly")
    return True


def run_all_tests():
    """Run all tests and provide summary"""
    print("\n" + "="*70)
    print("TRINETBSRNN INTEGRATION TEST SUITE")
    print("="*70)
    print("Testing integration of Trinet network with BSRNN pipeline")
    print("="*70)

    tests = [
        ("Shape Compatibility", test_1_shape_compatibility),
        ("Novel Components", test_2_novel_components),
        ("Forward Pass", test_3_forward_pass),
        ("Gradient Flow", test_4_gradient_flow),
        ("BSRNN Pipeline", test_5_bsrnn_training_compatibility),
        ("BSRNN Alias", test_6_alias_verification),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED", None))
        except Exception as e:
            results.append((test_name, "FAILED", str(e)))
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, status, error in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)

    print("="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Integration is correct.")
        print("\nThe Trinet network is successfully integrated with BSRNN pipeline:")
        print("  ✓ Novel components (FAC, AIA_Transformer) are preserved")
        print("  ✓ Input/output formats are compatible")
        print("  ✓ Can be used as drop-in replacement for BSRNN")
        print("  ✓ Training loop will work without modifications")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
