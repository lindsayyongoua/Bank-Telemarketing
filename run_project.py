#!/usr/bin/env python
"""
=================================================================================
AUTOMATED SETUP AND RUNNER SCRIPT
=================================================================================
This script helps set up and run the bank marketing prediction system.
Usage: python run_project.py [command]
Commands: eda, streamlit, both, check
=================================================================================
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.END}\n")

def print_step(text):
    """Print formatted step"""
    print(f"{Colors.CYAN}➜{Colors.END} {Colors.BOLD}{text}{Colors.END}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {Colors.BOLD}{text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {Colors.BOLD}{text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {Colors.BOLD}{text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")

def check_python_version():
    """Check if Python version is 3.8+"""
    print_step("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.8+ required. Found {version.major}.{version.minor}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_step("Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'streamlit': 'streamlit'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print_success(f"{package_name} is installed")
        except ImportError:
            print_warning(f"{package_name} is NOT installed")
            missing.append(package_name)
    
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        print_info("Run: pip install -r requirements.txt")
        return False
    
    print_success("All dependencies are installed!")
    return True

def check_data_files():
    """Check if required data files exist"""
    print_step("Checking data files...")
    
    required_files = [
        'bank.csv'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size / 1024 / 1024
            print_success(f"{file} found ({size:.2f} MB)")
        else:
            print_warning(f"{file} NOT found")
            missing.append(file)
    
    if missing:
        print_error(f"Missing files: {', '.join(missing)}")
        return False
    
    return True

def check_model_files():
    """Check if trained model files exist"""
    print_step("Checking model files...")
    
    model_files = [
        'best_model.pkl',
        'label_encoders.pkl',
        'df_model_preprocessed.csv'
    ]
    
    all_exist = True
    for file in model_files:
        if Path(file).exists():
            size = Path(file).stat().st_size / 1024
            print_success(f"{file} found ({size:.2f} KB)")
        else:
            print_warning(f"{file} NOT found - Run EDA to generate")
            all_exist = False
    
    return all_exist

def check_system():
    """Run all system checks"""
    print_header("🔍 SYSTEM CHECK")
    
    print_info(f"Operating System: {platform.system()} {platform.release()}")
    print_info(f"Python Executable: {sys.executable}")
    print()
    
    checks_passed = 0
    checks_total = 4
    
    if check_python_version():
        checks_passed += 1
    
    print()
    if check_dependencies():
        checks_passed += 1
    
    print()
    if check_data_files():
        checks_passed += 1
    
    print()
    if check_model_files():
        checks_passed += 1
    
    print()
    print_header(f"✓ System Check: {checks_passed}/{checks_total} passed")
    
    return checks_passed == checks_total

def run_eda():
    """Run exploratory data analysis"""
    print_header("📊 RUNNING EXPLORATORY DATA ANALYSIS")
    
    if not Path('bank.csv').exists():
        print_error("bank.csv not found!")
        return False
    
    try:
        print_step("Starting EDA analysis...")
        print_info("This may take 2-5 minutes on first run...\n")
        
        result = subprocess.run(
            [sys.executable, 'eda_analysis.py'],
            capture_output=False
        )
        
        if result.returncode == 0:
            print_header("✓ EDA COMPLETED SUCCESSFULLY!")
            
            # Check for generated files
            generated_files = [
                f"0{i}_*.png" for i in range(1, 11)
            ] + [
                'best_model.pkl',
                'label_encoders.pkl',
                'df_model_preprocessed.csv'
            ]
            
            print_step("Generated files:")
            for file in sorted(Path('.').glob('0[0-9]_*.png')):
                print_success(f"  {file.name}")
            
            model_files = ['best_model.pkl', 'label_encoders.pkl', 'df_model_preprocessed.csv']
            for file in model_files:
                if Path(file).exists():
                    print_success(f"  {file}")
            
            return True
        else:
            print_error("EDA failed with return code:", result.returncode)
            return False
    
    except Exception as e:
        print_error(f"Error running EDA: {str(e)}")
        return False

def run_streamlit():
    """Run Streamlit dashboard"""
    print_header("🎨 LAUNCHING STREAMLIT DASHBOARD")
    
    # Check if model files exist
    model_files = ['best_model.pkl', 'label_encoders.pkl', 'df_model_preprocessed.csv']
    missing = [f for f in model_files if not Path(f).exists()]
    
    if missing:
        print_error(f"Model files not found: {', '.join(missing)}")
        print_info("Run EDA first: python run_project.py eda")
        return False
    
    try:
        print_step("Starting Streamlit server...")
        print_info("Press CTRL+C to stop the server\n")
        print_success("Dashboard will open at: http://localhost:8501")
        print_info("If it doesn't open automatically, visit the URL manually\n")
        
        # Determine correct command based on OS
        if platform.system() == 'Windows':
            os.system('streamlit run app.py')
        else:
            subprocess.run(['streamlit', 'run', 'app.py'])
        
        return True
    
    except KeyboardInterrupt:
        print_warning("\nStreamlit server stopped by user")
        return True
    except Exception as e:
        print_error(f"Error running Streamlit: {str(e)}")
        return False

def install_dependencies():
    """Install required packages"""
    print_header("📦 INSTALLING DEPENDENCIES")
    
    try:
        print_step("Upgrading pip...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            capture_output=True
        )
        print_success("pip upgraded")
        
        if Path('requirements.txt').exists():
            print_step("Installing packages from requirements.txt...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                capture_output=False
            )
            
            if result.returncode == 0:
                print_success("All dependencies installed!")
                return True
            else:
                print_error("Failed to install dependencies")
                return False
        else:
            print_error("requirements.txt not found!")
            return False
    
    except Exception as e:
        print_error(f"Error installing dependencies: {str(e)}")
        return False

def show_menu():
    """Display main menu"""
    print_header("🏦 BANK MARKETING PREDICTION SYSTEM")
    
    print(f"{Colors.BOLD}Select an option:{Colors.END}\n")
    print("  1. {:<40} Check system and dependencies".format("check"))
    print("  2. {:<40} Install dependencies".format("install"))
    print("  3. {:<40} Run EDA Analysis".format("eda"))
    print("  4. {:<40} Launch Streamlit Dashboard".format("streamlit"))
    print("  5. {:<40} Run Both EDA and Dashboard".format("both"))
    print("  6. {:<40} Exit".format("exit"))
    print()
    
    choice = input(f"{Colors.CYAN}Enter your choice (1-6): {Colors.END}").strip()
    return choice

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = show_menu()
    
    if command in ['1', 'check']:
        check_system()
    
    elif command in ['2', 'install']:
        install_dependencies()
    
    elif command in ['3', 'eda']:
        if not check_dependencies():
            print_warning("Installing missing dependencies...")
            install_dependencies()
        
        run_eda()
    
    elif command in ['4', 'streamlit']:
        if not check_dependencies():
            print_warning("Installing missing dependencies...")
            install_dependencies()
        
        run_streamlit()
    
    elif command in ['5', 'both']:
        if not check_dependencies():
            print_warning("Installing missing dependencies...")
            install_dependencies()
        
        if run_eda():
            input(f"\n{Colors.CYAN}Press ENTER to start Streamlit dashboard...{Colors.END}")
            run_streamlit()
    
    elif command in ['6', 'exit', 'quit']:
        print_header("👋 GOODBYE!")
        sys.exit(0)
    
    else:
        print_error(f"Unknown command: {command}")
        print_info("Valid commands: check, install, eda, streamlit, both, exit")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Program interrupted by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)
