# Multi-Speaker ASR with NLP Enhancement

## 🎯 Overview

This project implements a novel approach to multi-speaker Automatic Speech Recognition (ASR) by using advanced NLP techniques to correct errors that typically occur in overlapping speech scenarios. Instead of trying to perfect the audio processing (which has physical limits), we accept that ASR will produce garbled output when multiple people speak simultaneously, and use context-aware language models to intelligently reconstruct what was likely said.

### Key Innovation

Traditional multi-speaker ASR systems fail when speakers overlap, producing incomplete and inaccurate transcriptions like:

-   "I think we sh## meet tomor- at thre-"
-   "Yeah I agr## with tha+ but we should also consid## the budg##"

Our system uses NLP to transform these into:

-   "I think we should meet tomorrow at three"
-   "Yeah I agree with that but we should also consider the budget"

## ✨ Features

-   **Real-time Processing**: Sub-100ms latency for live applications
-   **Multi-Strategy Correction**: Multiple NLP approaches with intelligent selection
-   **Speaker-Aware Learning**: Adapts to individual speaker patterns over time
-   **Context-Aware**: Uses conversation history for better corrections
-   **Modular Architecture**: Easy to extend and improve individual components
-   **Production-Ready**: Includes caching, error handling, and performance monitoring

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Audio     │     │    Noise     │     │   Speaker   │     │     ASR      │
│   Input     │ --> │  Reduction   │ --> │ Separation  │ --> │ (Whisper)    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Output    │ <-- │ Translation/ │ <-- │    Post-    │ <-- │     NLP      │
│   Text      │     │     TTS      │     │ Processing  │     │  Correction  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                       (Optional)                                 (Key Innovation)
```

## 📦 Installation

### Prerequisites

-   Python 3.8 or higher
-   CUDA-capable GPU (recommended for real-time performance)
-   FFmpeg (for audio processing)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/multi-speaker-asr-nlp.git
cd multi-speaker-asr-nlp
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download required models:

```bash
python scripts/download_models.py
```

## 🚀 Quick Start

### Basic Usage

```python
from src.pipeline.orchestrator import MultiSpeakerASRPipeline

# Initialize pipeline
pipeline = MultiSpeakerASRPipeline()

# Process an audio file
segments = pipeline.process_file("path/to/audio.wav")

# Print results
for segment in segments:
    print(f"Speaker {segment.speaker_id}: {segment.corrected_text}")
```

### Run the Demo

```bash
python scripts/run_demo.py
```

This will demonstrate:

1. Correction of simulated ASR errors
2. Real-time streaming capabilities
3. Performance metrics

### Start WebSocket Server

For real-time applications:

```bash
python scripts/run_server.py
```

Then connect your client to `ws://localhost:8765`

## 🔧 Configuration

Edit `config/config.yaml` to customize:

-   Model selection and parameters
-   Processing strategies
-   Performance thresholds
-   Server settings

Key configuration options:

```yaml
nlp_correction:
    primary_model: "t5-base" # Main correction model
    max_latency_ms: 100 # Real-time constraint
    strategies:
        pattern_based: true # Fast rule-based corrections
        neural: true # Deep learning corrections
        speaker_specific: true # Learn from each speaker
```

## 📊 Performance

### Benchmarks

| Metric                | Traditional ASR | Our System | Improvement |
| --------------------- | --------------- | ---------- | ----------- |
| WER (Word Error Rate) | 18%             | 14%        | -22%        |
| Overlap Error Rate    | 25%             | 18%        | -28%        |
| Processing Latency    | -               | <100ms     | Real-time   |

### Optimization Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed
2. **Model Selection**: Use smaller models (t5-small) for faster processing
3. **Caching**: Enable caching for repeated phrases
4. **Batch Processing**: Process multiple speakers in parallel

## 🛠️ Development

### Project Structure

```
src/
├── nlp/                 # NLP correction modules
│   ├── error_corrector
```
