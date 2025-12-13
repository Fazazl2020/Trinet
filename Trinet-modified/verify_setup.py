#!/usr/bin/env python
# coding: utf-8

"""
Pre-training verification script.
Checks that all dependencies, paths, and configurations are correct.
"""

import os
import sys

def check_python_version():
    """Check Python version."""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.7+)")
        return False

def check_imports():
    """Check required packages."""
    print("\nChecking required packages:")

    packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'numpy': 'NumPy',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'pystoi': 'PySTOI (for STOI computation)',
        'pesq': 'PESQ',
        'natsort': 'Natsort',
        'tqdm': 'TQDM',
        'joblib': 'Joblib'
    }

    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - Install with: pip install {package}")
            all_ok = False

    return all_ok

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA:")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"  ✅ CUDA available with {device_count} GPU(s)")
            for i in range(device_count):
                print(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("  ❌ CUDA not available (training will be very slow on CPU)")
            return False
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False

def check_files():
    """Check required files exist."""
    print("\nChecking required files:")

    required_files = [
        'train_improved.py',
        'evaluation_improved.py',
        'resume_training.py',
        'module.py',
        'dataloader.py',
        'utils.py',
        'IMPROVED_TRAINING_GUIDE.md'
    ]

    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - Missing!")
            all_ok = False

    return all_ok

def check_data_paths():
    """Check data directories."""
    print("\nChecking data paths:")

    # Try to import config
    try:
        from train_improved import args

        # Check data directory
        if os.path.exists(args.data_dir):
            print(f"  ✅ Data directory: {args.data_dir}")

            # Check train/test subdirectories
            train_dir = os.path.join(args.data_dir, 'train')
            test_dir = os.path.join(args.data_dir, 'test')

            if os.path.exists(train_dir):
                train_noisy = os.path.join(train_dir, 'noisy')
                train_clean = os.path.join(train_dir, 'clean')
                if os.path.exists(train_noisy) and os.path.exists(train_clean):
                    n_train_noisy = len(os.listdir(train_noisy))
                    n_train_clean = len(os.listdir(train_clean))
                    print(f"     - Train set: {n_train_noisy} noisy, {n_train_clean} clean files")
                    if n_train_noisy != n_train_clean:
                        print(f"     ⚠️  Warning: Mismatched train file counts!")
                else:
                    print(f"     ❌ Missing train/noisy or train/clean subdirectories")
            else:
                print(f"     ❌ Train directory not found: {train_dir}")

            if os.path.exists(test_dir):
                test_noisy = os.path.join(test_dir, 'noisy')
                test_clean = os.path.join(test_dir, 'clean')
                if os.path.exists(test_noisy) and os.path.exists(test_clean):
                    n_test_noisy = len(os.listdir(test_noisy))
                    n_test_clean = len(os.listdir(test_clean))
                    print(f"     - Test set: {n_test_noisy} noisy, {n_test_clean} clean files")
                    if n_test_noisy != n_test_clean:
                        print(f"     ⚠️  Warning: Mismatched test file counts!")
                else:
                    print(f"     ❌ Missing test/noisy or test/clean subdirectories")
            else:
                print(f"     ❌ Test directory not found: {test_dir}")

            return True
        else:
            print(f"  ❌ Data directory not found: {args.data_dir}")
            print(f"     Please edit train_improved.py line 30 to set correct path")
            return False

    except Exception as e:
        print(f"  ❌ Error checking data paths: {e}")
        return False

def check_save_directory():
    """Check save directory."""
    print("\nChecking checkpoint save directory:")

    try:
        from train_improved import args

        # Check if directory exists, create if not
        if os.path.exists(args.save_model_dir):
            print(f"  ✅ Save directory exists: {args.save_model_dir}")

            # Check if writable
            test_file = os.path.join(args.save_model_dir, '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"     - Directory is writable ✅")
                return True
            except:
                print(f"     ❌ Directory is not writable!")
                return False
        else:
            print(f"  ℹ️  Save directory does not exist: {args.save_model_dir}")
            print(f"     Will be created automatically during training")

            # Try to create it
            try:
                os.makedirs(args.save_model_dir, exist_ok=True)
                print(f"     ✅ Successfully created directory")
                return True
            except Exception as e:
                print(f"     ❌ Cannot create directory: {e}")
                return False

    except Exception as e:
        print(f"  ❌ Error checking save directory: {e}")
        return False

def check_model_initialization():
    """Check model can be initialized."""
    print("\nChecking model initialization:")

    try:
        import torch
        from module import BSRNN, Discriminator

        # Try to create model
        model = BSRNN(num_channel=64, num_layer=5)
        disc = Discriminator(ndf=16)

        # Check parameter counts
        n_params = sum(p.numel() for p in model.parameters())
        n_disc_params = sum(p.numel() for p in disc.parameters())

        print(f"  ✅ Model initialized successfully")
        print(f"     - Generator parameters: {n_params:,}")
        print(f"     - Discriminator parameters: {n_disc_params:,}")

        # Try CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            disc_cuda = disc.cuda()
            print(f"     - CUDA transfer successful ✅")

        return True

    except Exception as e:
        print(f"  ❌ Error initializing model: {e}")
        return False

def print_config():
    """Print current configuration."""
    print("\n" + "="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)

    try:
        from train_improved import args

        print(f"Training epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Initial learning rate: {args.init_lr}")
        print(f"Cut length: {args.cut_len / 16000:.1f} seconds")
        print(f"\nLoss weights:")
        print(f"  RI: {args.loss_weights['ri']}")
        print(f"  Magnitude: {args.loss_weights['mag']}")
        print(f"  Time: {args.loss_weights['time']}")
        print(f"  Discriminator: {args.loss_weights['disc']}")
        print(f"\nDiscriminator configuration:")
        print(f"  Start ratio: {args.disc_start_ratio} (epoch {int(args.epochs * args.disc_start_ratio)})")
        print(f"  Warmup epochs: {args.disc_warmup_epochs}")
        print(f"\nComposite score weights:")
        print(f"  PESQ: {args.composite_weights['pesq']}")
        print(f"  STOI: {args.composite_weights['stoi']}")
        print(f"  Loss: {args.composite_weights['loss']}")
        print(f"\nPaths:")
        print(f"  Data: {args.data_dir}")
        print(f"  Checkpoints: {args.save_model_dir}")

    except Exception as e:
        print(f"Error loading configuration: {e}")

def main():
    """Run all checks."""
    print("="*70)
    print("IMPROVED TRAINING SETUP VERIFICATION")
    print("="*70)

    checks = [
        ("Python version", check_python_version),
        ("Required packages", check_imports),
        ("CUDA availability", check_cuda),
        ("Required files", check_files),
        ("Data paths", check_data_paths),
        ("Save directory", check_save_directory),
        ("Model initialization", check_model_initialization),
    ]

    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))

    # Print configuration
    print_config()

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<30} {status}")

    print("="*70)

    if all_passed:
        print("\n✅ ALL CHECKS PASSED - Ready to train!")
        print("\nNext steps:")
        print("  1. Review configuration above")
        print("  2. Run: python train_improved.py")
        print("  3. Monitor training logs")
        print("  4. Evaluate with: python evaluation_improved.py")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED - Please fix issues above before training")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install <package_name>")
        print("  - Edit train_improved.py lines 30-31 to set correct paths")
        print("  - Ensure CUDA/PyTorch is properly installed")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
