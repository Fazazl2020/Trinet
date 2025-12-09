"""
Syntax Validation Script for Ablation Study Models

Validates Python syntax without requiring torch installation.
"""

import ast
import os
import sys

def validate_syntax(model_name, file_path):
    """Validate Python syntax of a module"""
    print(f"\n{'='*70}")
    print(f"VALIDATING SYNTAX: {model_name}")
    print(f"{'='*70}")

    try:
        # Check file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False

        print(f"[1/3] Reading file...")
        with open(file_path, 'r') as f:
            code = f.read()
        print(f"✓ File read successfully ({len(code)} characters)")

        # Parse AST
        print(f"[2/3] Parsing Python syntax...")
        ast.parse(code)
        print(f"✓ Syntax valid")

        # Check for required classes
        print(f"[3/3] Checking required classes...")
        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        required_classes = ['TrinetBSRNN', 'Discriminator', 'BSRNN']
        found = []
        missing = []

        for cls in required_classes:
            if cls in classes:
                found.append(cls)
            else:
                missing.append(cls)

        if missing:
            print(f"⚠️  Missing classes: {missing}")
            if 'BSRNN' in missing and 'TrinetBSRNN' in found:
                print(f"   Note: BSRNN should be aliased to TrinetBSRNN")

        print(f"✓ Found classes: {found}")

        print(f"\n{'='*70}")
        print(f"✅ {model_name} SYNTAX VALIDATION PASSED")
        print(f"{'='*70}")
        return True

    except SyntaxError as e:
        print(f"\n{'='*70}")
        print(f"❌ {model_name} SYNTAX ERROR")
        print(f"{'='*70}")
        print(f"Line {e.lineno}: {e.msg}")
        print(f"Text: {e.text}")
        return False
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ {model_name} VALIDATION FAILED")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        return False

def main():
    """Run syntax validation on all ablation models"""
    print("\n" + "="*70)
    print("ABLATION STUDY MODEL SYNTAX VALIDATION")
    print("="*70)
    print("\nNote: This validates Python syntax only.")
    print("Full validation (with torch) should be done on your training server.")

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
        results[model_name] = validate_syntax(model_name, full_path)

    # Summary
    print("\n" + "="*70)
    print("SYNTAX VALIDATION SUMMARY")
    print("="*70)

    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {model_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL MODELS SYNTAX VALIDATED!")
        print("="*70)
        print("\nNext steps:")
        print("1. Copy the following files to each model directory:")
        print("   - train.py")
        print("   - dataloader.py")
        print("   - utils.py")
        print("   - evaluation.py")
        print("\n2. Modify save_model_dir in each train.py:")
        print("   M1: '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M1_Baseline/ckpt'")
        print("   M2: '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M2_FAC/ckpt'")
        print("   M3: '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M3_FAC_SingleMRHA/ckpt'")
        print("   M4: '/ghome/fewahab/My_5th_pap/Ab4-BSRNN/M4_FAC_FullMRHA/ckpt'")
        print("\n3. On your server, run full validation:")
        print("   cd ablation_models && python validate_models.py")
        print("\n4. Start training:")
        print("   python train.py")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️  SOME MODELS FAILED SYNTAX VALIDATION")
        print("="*70)
        print("\nPlease review the errors above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
