"""
Validation Script for Ablation Study Models

This script validates that all ablation models:
1. Import correctly
2. Instantiate without errors
3. Forward pass works
4. Output shapes are correct
5. No NaN/Inf values in outputs
"""

import sys
import os
import torch
import importlib.util

def load_module_from_path(module_name, file_path):
    """Dynamically load module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def validate_model(model_name, model_path):
    """Validate a single ablation model"""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {model_name}")
    print(f"{'='*70}")

    try:
        # Step 1: Import module
        print(f"[1/6] Importing module from {model_path}...")
        module = load_module_from_path(f"module_{model_name}", model_path)
        print("✓ Module imported successfully")

        # Step 2: Check required classes exist
        print("[2/6] Checking required classes...")
        assert hasattr(module, 'BSRNN'), "BSRNN class not found"
        assert hasattr(module, 'Discriminator'), "Discriminator class not found"
        print("✓ Required classes found")

        # Step 3: Instantiate model
        print("[3/6] Instantiating model (num_channel=64, num_layer=5)...")
        model = module.BSRNN(num_channel=64, num_layer=5, F=257)
        discriminator = module.Discriminator(ndf=16)
        print(f"✓ Model instantiated successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

        # Step 4: Create test input
        print("[4/6] Creating test input...")
        batch_size = 2
        F_bins = 257
        T_frames = 251

        # Complex spectrogram input [B, F, T]
        test_input = torch.randn(batch_size, F_bins, T_frames, dtype=torch.complex64)
        print(f"✓ Test input created: {test_input.shape}")

        # Step 5: Forward pass
        print("[5/6] Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass successful")
        print(f"   Input shape:  {test_input.shape}")
        print(f"   Output shape: {output.shape}")

        # Step 6: Validate output
        print("[6/6] Validating output...")

        # Check shape
        assert output.shape == test_input.shape, f"Shape mismatch! Expected {test_input.shape}, got {output.shape}"
        print(f"✓ Output shape matches input")

        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN values!"
        assert not torch.isinf(output).any(), "Output contains Inf values!"
        print(f"✓ No NaN/Inf values detected")

        # Check dtype
        assert output.dtype == torch.complex64, f"Output dtype should be complex64, got {output.dtype}"
        print(f"✓ Output dtype correct (complex64)")

        print(f"\n{'='*70}")
        print(f"✅ {model_name} VALIDATION PASSED")
        print(f"{'='*70}")
        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ {model_name} VALIDATION FAILED")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run validation on all ablation models"""
    print("\n" + "="*70)
    print("ABLATION STUDY MODEL VALIDATION")
    print("="*70)

    # Define models to validate
    models = [
        ("M1: Conv2D + Standard Transformer",
         "M1_Conv2D_StandardTransformer/module.py"),
        ("M2: FAC + Standard Transformer",
         "M2_FAC_StandardTransformer/module.py"),
        ("M3: FAC + Single-Branch MRHA",
         "M3_FAC_SingleBranchMRHA/module.py"),
        ("M4: FAC + Full MRHA (Proposed)",
         "M4_FAC_FullMRHA/module.py"),
    ]

    results = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for model_name, model_path in models:
        full_path = os.path.join(base_dir, model_path)

        if not os.path.exists(full_path):
            print(f"\n❌ {model_name}: File not found at {full_path}")
            results[model_name] = False
            continue

        results[model_name] = validate_model(model_name, full_path)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {model_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL MODELS VALIDATED SUCCESSFULLY!")
        print("="*70)
        print("\nYou can now proceed with training:")
        print("1. Copy train.py, dataloader.py, utils.py to each model directory")
        print("2. Modify save_model_dir in each train.py")
        print("3. Run: python train.py")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️  SOME MODELS FAILED VALIDATION")
        print("="*70)
        print("\nPlease fix the errors before training.")
        print("="*70)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
