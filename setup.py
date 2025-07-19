#!/usr/bin/env python3
"""
Setup script for ICP Project Evaluation System
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        sys.exit(1)

def setup_environment():
    """Set up environment file."""
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✓ Environment file {env_file} already exists")
        return
    
    print("Setting up environment file...")
    try:
        with open("config.env.example", "r") as f:
            content = f.read()
        
        with open(env_file, "w") as f:
            f.write(content)
        
        print(f"✓ Created {env_file} file")
        print("⚠️  Please edit .env file and add your API keys:")
        print("   - GROQ_API_KEY: Get from https://console.groq.com/")
        print("   - GITHUB_TOKEN: Get from GitHub Settings → Developer settings → Personal access tokens")
    except Exception as e:
        print(f"✗ Failed to create environment file: {e}")
        sys.exit(1)

def main():
    print("ICP Project Evaluation System - Setup")
    print("=" * 40)
    
    check_python_version()
    install_dependencies()
    setup_environment()
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Test the system: python test_evaluator.py")
    print("3. Run evaluation: python main.py sample_input.csv results.csv")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 