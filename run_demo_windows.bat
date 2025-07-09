@echo off
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set SPEECHBRAIN_USE_SYMLINKS=0
set TRANSFORMERS_CACHE=./cache/transformers
set HF_HOME=./cache/huggingface

echo Running demo with Windows-specific settings...
python scripts/run_demo.py %*
