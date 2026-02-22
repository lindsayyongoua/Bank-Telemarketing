#!/usr/bin/env python
"""
=================================================================================
TEST SCRIPT - VERIFY PROJECT SETUP
=================================================================================
This script tests all components are working correctly.
Run: python test_project.py
=================================================================================
"""

import sys
import os
from pathlib import Path
import subprocess

def test_imports():
    """Test if all required packages can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Checking Python Imports")
    print("="*60)
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'streamlit': 'streamlit',
        'pickle': 'pickle (built-in)'
    }
    
    all_passed = True
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name:30} OK")
        except ImportError as e:
            print(f"✗ {package_name:30} MISSING - Install: pip install {package_name}")
            all_passed = False
    
    return all_passed

def test_data_files():
    """Test if data files exist"""
    print("\n" + "="*60)
    print("TEST 2: Checking Data Files")
    print("="*60)
    
    files = {
        'bank.csv': 'Main dataset (required)',
        'bank-full.csv': 'Full dataset (optional)',
        'bank-names.txt': 'Feature descriptions (optional)'
    }
    
    all_passed = True
    for filename, description in files.items():
        if Path(filename).exists():
            size = Path(filename).stat().st_size / 1024 / 1024
            print(f"✓ {filename:30} ({size:.2f} MB) - {description}")
        else:
            print(f"✗ {filename:30} NOT FOUND - {description}")
            if 'required' in description:
                all_passed = False
    
    return all_passed

def test_eda_script():
    """Test if EDA script exists and has no syntax errors"""
    print("\n" + "="*60)
    print("TEST 3: Checking EDA Script")
    print("="*60)
    
    if not Path('eda_analysis.py').exists():
        print("✗ eda_analysis.py NOT FOUND")
        return False
    
    try:
        with open('eda_analysis.py', 'r') as f:
            code = f.read()
        compile(code, 'eda_analysis.py', 'exec')
        print(f"✓ eda_analysis.py exists ({len(code)} bytes)")
        print(f"✓ No syntax errors detected")
        
        # Count sections
        sections = code.count('# SECTION')
        print(f"✓ Found {sections} analysis sections")
        
        # Count visualizations
        plots = code.count('plt.savefig')
        print(f"✓ Found {plots} visualization outputs")
        
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in eda_analysis.py: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app exists and has no syntax errors"""
    print("\n" + "="*60)
    print("TEST 4: Checking Streamlit App")
    print("="*60)
    
    if not Path('app.py').exists():
        print("✗ app.py NOT FOUND")
        return False
    
    try:
        with open('app.py', 'r') as f:
            code = f.read()
        compile(code, 'app.py', 'exec')
        print(f"✓ app.py exists ({len(code)} bytes)")
        print(f"✓ No syntax errors detected")
        
        # Count pages
        pages = code.count('if page ==')
        print(f"✓ Found {pages} dashboard pages")
        
        # Check for custom styling
        if '<style>' in code:
            print(f"✓ Custom CSS styling included")
        
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in app.py: {e}")
        return False

def test_requirements():
    """Test if requirements.txt is present"""
    print("\n" + "="*60)
    print("TEST 5: Checking Requirements File")
    print("="*60)
    
    if not Path('requirements.txt').exists():
        print("✗ requirements.txt NOT FOUND")
        return False
    
    with open('requirements.txt', 'r') as f:
        packages = f.readlines()
    
    print(f"✓ requirements.txt exists")
    print(f"✓ Found {len(packages)} package dependencies:")
    for pkg in packages:
        pkg = pkg.strip()
        if pkg and not pkg.startswith('#'):
            print(f"  - {pkg}")
    
    return True

def test_documentation():
    """Test if documentation files exist"""
    print("\n" + "="*60)
    print("TEST 6: Checking Documentation")
    print("="*60)
    
    docs = {
        'README.md': 'Main documentation',
        'DEPLOYMENT_GUIDE.md': 'Deployment instructions',
        'QUICKSTART.md': 'Quick reference',
        'COMPLETION_SUMMARY.md': 'Completion summary',
        '.gitignore': 'Git ignore file'
    }
    
    all_found = True
    for filename, description in docs.items():
        if Path(filename).exists():
            size = Path(filename).stat().st_size / 1024
            print(f"✓ {filename:30} ({size:.2f} KB) - {description}")
        else:
            print(f"✗ {filename:30} NOT FOUND - {description}")
            all_found = False
    
    return all_found

def test_model_files():
    """Test if model files exist (if EDA has been run)"""
    print("\n" + "="*60)
    print("TEST 7: Checking Model Files (Optional - Run EDA First)")
    print("="*60)
    
    model_files = {
        'best_model.pkl': 'Trained model',
        'label_encoders.pkl': 'Feature encoders',
        'df_model_preprocessed.csv': 'Processed data'
    }
    
    found = 0
    for filename, description in model_files.items():
        if Path(filename).exists():
            size = Path(filename).stat().st_size / 1024
            print(f"✓ {filename:30} ({size:.2f} KB) - {description}")
            found += 1
        else:
            print(f"○ {filename:30} NOT FOUND - {description}")
            print(f"  (Will be generated after running: python eda_analysis.py)")
    
    if found == 0:
        print("\nℹ️  Model files will be created after running EDA analysis")
    
    return True

def test_visualization_files():
    """Test if visualization files exist (if EDA has been run)"""
    print("\n" + "="*60)
    print("TEST 8: Checking Visualization Files (Optional - Run EDA First)")
    print("="*60)
    
    viz_files = [
        '01_target_distribution.png',
        '02_numerical_distributions.png',
        '03_outliers_boxplots.png',
        '04_categorical_distributions.png',
        '05_numerical_vs_target.png',
        '06_categorical_vs_target.png',
        '07_correlation_matrix.png',
        '08_feature_importance.png',
        '09_models_comparison.png',
        '10_confusion_matrices.png'
    ]
    
    found = 0
    for filename in viz_files:
        if Path(filename).exists():
            size = Path(filename).stat().st_size / 1024
            print(f"✓ {filename:40} ({size:.2f} KB)")
            found += 1
        else:
            print(f"○ {filename:40} NOT FOUND")
    
    if found == 0:
        print("\nℹ️  Visualization files will be created after running EDA analysis")
    else:
        print(f"\n✓ Generated {found}/10 visualization files")
    
    return True

def test_directory_structure():
    """Test if directory structure is correct"""
    print("\n" + "="*60)
    print("TEST 9: Checking Directory Structure")
    print("="*60)
    
    directories = ['.streamlit', 'src']
    
    for dirname in directories:
        if Path(dirname).exists():
            print(f"✓ {dirname}/ directory exists")
        else:
            print(f"○ {dirname}/ directory not found (optional)")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" "*20 + "🏦 BANK MARKETING PROJECT - TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Run tests
    results['Imports'] = test_imports()
    results['Data Files'] = test_data_files()
    results['EDA Script'] = test_eda_script()
    results['Streamlit App'] = test_streamlit_app()
    results['Requirements'] = test_requirements()
    results['Documentation'] = test_documentation()
    results['Model Files'] = test_model_files()
    results['Visualization Files'] = test_visualization_files()
    results['Directory Structure'] = test_directory_structure()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} {test_name}")
    
    print("\n" + "="*60)
    
    if passed == total:
        print("✓ ALL TESTS PASSED - Project is ready!")
        print("\nNext steps:")
        print("  1. Run EDA analysis: python run_project.py eda")
        print("  2. Launch dashboard: python run_project.py streamlit")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED - Please fix issues above")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check data files exist")
        print("  3. Verify Python 3.8+: python --version")
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)



