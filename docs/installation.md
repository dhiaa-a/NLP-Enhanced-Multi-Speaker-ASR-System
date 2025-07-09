# Installation Guide

This guide will help you install and set up the Multi-Speaker ASR with NLP Enhancement system.

## Prerequisites

Before installing, ensure you have:

-   Python 3.8 or higher
-   pip (Python package manager)
-   git
-   At least 8GB RAM
-   (Optional) CUDA-capable GPU for faster processing

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-speaker-asr-nlp.git
cd multi-speaker-asr-nlp
```

### 2. Create Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download Models

```bash
# Download required pre-trained models
python models/download_models.py
```

This will download:

-   NLP models (T5, BERT) from Hugging Face
-   Whisper ASR model
-   Other required models

## Detailed Installation

### System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev
```

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.9 ffmpeg portaudio
```

#### Windows

1. Install Python 3.8+ from [python.org](https://www.python.org)
2. Install FFmpeg:
    - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
    - Add to PATH
3. Install Visual Studio Build Tools for C++ dependencies

### Python Dependencies

#### Core Requirements

```bash
pip install -r requirements.txt
```

This installs:

-   PyTorch and torchaudio
-   Transformers (Hugging Face)
-   OpenAI Whisper
-   Audio processing libraries
-   Web server components

#### Development Dependencies

```bash
pip install -r requirements-dev.txt
```

Includes:

-   pytest for testing
-   black for code formatting
-   mypy for type checking

### GPU Setup (Optional)

#### NVIDIA GPU

1. Install CUDA Toolkit 11.8 or higher:

    ```bash
    # Check CUDA version
    nvidia-smi
    ```

2. Install PyTorch with CUDA:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

3. Verify GPU is detected:
    ```python
    import torch
    print(torch.cuda.is_available())  # Should print True
    print(torch.cuda.get_device_name())
    ```

## Model Setup

### Automatic Download

Run the model download script:

```bash
python models/download_models.py
```

Options:

-   `--verify-only`: Check if models are already downloaded
-   `--models-dir`: Specify custom model directory

### Manual Download

If automatic download fails:

1. **Whisper Models**:

    ```python
    import whisper
    model = whisper.load_model("base")  # Downloads automatically
    ```

2. **Hugging Face Models**:
    ```python
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    ```

## Configuration

### Basic Configuration

Edit `config/config.yaml`:

```yaml
# Adjust based on your system
nlp_correction:
    primary_model: "t5-base" # or "t5-small" for faster processing
    max_latency_ms: 100 # Increase for better accuracy

optimization:
    use_gpu: true # Set to false if no GPU
    batch_size: 4 # Reduce if running out of memory
```

### Environment Variables

Create `.env` file:

```bash
# Optional environment variables
CUDA_VISIBLE_DEVICES=0  # Specify GPU
TRANSFORMERS_CACHE=/path/to/cache  # Model cache directory
```

## Verification

### Test Installation

Run the test script:

```bash
python -c "from src import MultiSpeakerASRPipeline; print('Installation successful!')"
```

### Run Tests

```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_nlp_correction.py
```

### Run Demo

```bash
# Run the demo script
python scripts/run_demo.py
```

## Troubleshooting

### Common Issues

#### 1. PyAudio Installation Fails

**Linux**:

```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**macOS**:

```bash
brew install portaudio
pip install pyaudio
```

**Windows**:
Download wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

#### 2. CUDA Out of Memory

Reduce batch size in config:

```yaml
optimization:
    batch_size: 1 # Reduce from 4
```

Or use CPU:

```yaml
optimization:
    use_gpu: false
```

#### 3. Model Download Fails

-   Check internet connection
-   Try manual download
-   Use smaller models (t5-small instead of t5-base)

#### 4. Import Errors

Ensure you're in the project root:

```bash
cd multi-speaker-asr-nlp
python scripts/run_demo.py
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Search [existing issues](https://github.com/yourusername/multi-speaker-asr-nlp/issues)
3. Create a new issue with:
    - Python version
    - OS version
    - Error message
    - Steps to reproduce

## Next Steps

-   Read the [Usage Guide](usage.md)
-   Try the [examples](../examples/basic_usage.py)
-   Run the [demo](../scripts/run_demo.py)
-   Start the [WebSocket server](../scripts/run_server.py)

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
python models/download_models.py --verify-only
```
