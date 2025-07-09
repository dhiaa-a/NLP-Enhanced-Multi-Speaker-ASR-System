#!/usr/bin/env python3
"""
Windows-specific setup script to fix common issues
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_windows_environment():
    """Setup Windows-specific environment variables and fixes"""
    
    print("Setting up Windows environment...")
    
    # 1. Disable symlinks warning for Hugging Face
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    # 2. Set SpeechBrain to not use symlinks
    os.environ['SPEECHBRAIN_USE_SYMLINKS'] = '0'
    
    # 3. Create .env file with these settings
    env_content = """# Windows-specific environment variables
HF_HUB_DISABLE_SYMLINKS_WARNING=1
SPEECHBRAIN_USE_SYMLINKS=0
TRANSFORMERS_CACHE=./cache/transformers
HF_HOME=./cache/huggingface
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✓ Created .env file with Windows settings")
    
    # 4. Create cache directories
    cache_dirs = ['cache/transformers', 'cache/huggingface', 'models/sepformer']
    for dir_path in cache_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created cache directories")
    
    # 5. Install Windows-specific dependencies
    print("\nInstalling Windows-specific dependencies...")
    
    # PyAudio wheel for Windows
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    if sys.maxsize > 2**32:  # 64-bit
        pyaudio_wheel = f"PyAudio-0.2.11-cp{python_version}-cp{python_version}-win_amd64.whl"
    else:  # 32-bit
        pyaudio_wheel = f"PyAudio-0.2.11-cp{python_version}-cp{python_version}-win32.whl"
    
    print(f"Note: If PyAudio installation fails, download {pyaudio_wheel} from:")
    print("https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
    
    # 6. Create a simple batch file to run with environment variables
    batch_content = """@echo off
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set SPEECHBRAIN_USE_SYMLINKS=0
set TRANSFORMERS_CACHE=./cache/transformers
set HF_HOME=./cache/huggingface

echo Running demo with Windows-specific settings...
python scripts/run_demo.py %*
"""
    
    with open('run_demo_windows.bat', 'w') as f:
        f.write(batch_content)
    
    print("✓ Created run_demo_windows.bat")
    
    print("\n✅ Windows setup complete!")
    print("\nTo run the demo on Windows, use:")
    print("  run_demo_windows.bat")
    print("\nOr set environment variables manually:")
    print("  set HF_HUB_DISABLE_SYMLINKS_WARNING=1")
    print("  set SPEECHBRAIN_USE_SYMLINKS=0")
    print("  python scripts/run_demo.py")

if __name__ == "__main__":
    setup_windows_environment()