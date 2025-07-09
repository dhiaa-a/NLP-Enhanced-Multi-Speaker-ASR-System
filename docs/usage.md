# Usage Guide

This guide covers how to use the Multi-Speaker ASR with NLP Enhancement system for various tasks.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Real-Time Processing](#real-time-processing)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Best Practices](#best-practices)

## Quick Start

### Process an Audio File

```python
from src import MultiSpeakerASRPipeline

# Initialize pipeline
pipeline = MultiSpeakerASRPipeline()

# Process audio file
segments = pipeline.process_file("path/to/audio.wav")

# Print results
for segment in segments:
    print(f"Speaker {segment.speaker_id}: {segment.corrected_text}")
```

### Correct Garbled Text

```python
from src import NLPErrorCorrector

# Initialize corrector
corrector = NLPErrorCorrector()

# Correct garbled ASR output
raw_text = "I think we sh## meet tomor- at thre-"
corrected = corrector.correct_text(raw_text)
print(corrected)  # "I think we should meet tomorrow at three"
```

## Basic Usage

### 1. Processing Audio Files

```python
from src.pipeline.orchestrator import MultiSpeakerASRPipeline

# Initialize with custom config
pipeline = MultiSpeakerASRPipeline(config_path="config/config.yaml")

# Process single file
segments = pipeline.process_file("meeting.wav")

# Access detailed information
for segment in segments:
    print(f"Speaker {segment.speaker_id}")
    print(f"Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
    print(f"Raw ASR: {segment.raw_text}")
    print(f"Corrected: {segment.corrected_text}")
    print(f"Confidence: {segment.confidence:.2f}")
    print()
```

### 2. NLP Error Correction

```python
from src.nlp.error_corrector import NLPErrorCorrector, TranscriptionSegment

# Initialize corrector
corrector = NLPErrorCorrector()

# Create segment with garbled text
segment = TranscriptionSegment(
    speaker_id=1,
    start_time=0.0,
    end_time=3.0,
    raw_text="The budg## is appro### five mill###"
)

# Correct with context
corrector.context_buffer = [
    "We need to discuss the financial plan",
    "Let's review the numbers"
]
segment.context_window = corrector.get_context_window()

# Apply correction
corrected_segment = corrector.correct_segment(segment)
print(corrected_segment.corrected_text)
# Output: "The budget is approximately five million"
```

### 3. Real-Time Correction

```python
from src.nlp.realtime_corrector import RealTimeErrorCorrector

# Initialize with 50ms deadline
realtime_corrector = RealTimeErrorCorrector(max_latency_ms=50)

# Correct with deadline
result = realtime_corrector.correct_with_deadline(
    "Quick fix for meet- tomorrow"
)

print(f"Corrected: {result.text}")
print(f"Latency: {result.latency_ms:.1f}ms")
print(f"Strategy used: {result.strategy}")
```

## Advanced Features

### 1. Speaker-Aware Correction

Learn from individual speakers for better accuracy:

```python
from src.nlp.speaker_models import SpeakerAwareCorrector

# Initialize speaker-aware corrector
speaker_corrector = SpeakerAwareCorrector()

# Train on speaker's utterances
speaker_id = 1
training_texts = [
    "I work on machine learning algorithms",
    "The neural network architecture is complex",
    "We need to optimize the model performance"
]

for text in training_texts:
    speaker_corrector.update_speaker_model(speaker_id, text)

# Correct with speaker context
corrupted = "We need to optim### the mod### perform###"
corrected = speaker_corrector.speaker_specific_correction(
    corrupted, speaker_id
)
print(corrected)  # Uses speaker's vocabulary
```

### 2. Streaming Audio Processing

Process audio in real-time:

```python
import asyncio
from src.pipeline.orchestrator import AudioSegment

async def process_stream():
    pipeline = MultiSpeakerASRPipeline()

    async def audio_generator():
        # Your audio stream source
        while True:
            audio_chunk = get_next_audio_chunk()  # Your implementation
            yield AudioSegment(
                data=audio_chunk,
                sample_rate=16000,
                timestamp=get_timestamp(),
                duration=0.5
            )

    async for result in pipeline.process_audio_stream(audio_generator()):
        for segment in result.segments:
            print(f"Speaker {segment.speaker_id}: {segment.corrected_text}")
```

### 3. Batch Processing

Process multiple texts efficiently:

```python
from src.nlp.realtime_corrector import RealTimeErrorCorrector

corrector = RealTimeErrorCorrector()

texts = [
    "Meeting at thre### tomorrow",
    "Budget approv### for Q4",
    "New proj### deadline Friday"
]

results = corrector.batch_correct(texts)

for text, result in zip(texts, results):
    print(f"{text} -> {result.text} ({result.latency_ms:.1f}ms)")
```

### 4. Custom Correction Strategies

Use specific correction strategies:

```python
from src.nlp.correction_strategies import CorrectionStrategies

strategies = CorrectionStrategies()

# Pattern-based correction (fastest)
corrected = strategies.pattern_based_correction("meet- tomorrow")

# Phonetic correction
corrected = strategies.phonetic_correction("thre people")

# Context-aware correction
context = ["Let's schedule a meeting", "When should we meet?"]
corrected = strategies.contextual_correction("tomor### at thre", context)

# Apply all strategies and compare
results = strategies.apply_all_strategies(
    "The budg## is ready",
    context=context
)
print(results)  # Dict with results from each strategy
```

## Real-Time Processing

### WebSocket Server

Start the WebSocket server:

```bash
python scripts/run_server.py --host 0.0.0.0 --port 8765
```

### JavaScript Client Example

```javascript
const ws = new WebSocket("ws://localhost:8765")

ws.onopen = () => {
	console.log("Connected to ASR server")

	// Send audio data
	const audioData = new Float32Array(16000) // 1 second
	const base64Audio = btoa(
		String.fromCharCode(...new Uint8Array(audioData.buffer)),
	)

	ws.send(
		JSON.stringify({
			type: "audio",
			audio: base64Audio,
			sample_rate: 16000,
			timestamp: Date.now() / 1000,
		}),
	)
}

ws.onmessage = (event) => {
	const data = JSON.parse(event.data)

	if (data.type === "transcription") {
		data.segments.forEach((segment) => {
			console.log(
				`Speaker ${segment.speaker_id}: ${segment.corrected_text}`,
			)
		})
	}
}
```

### Python Client Example

```python
import websockets
import asyncio
import json
import numpy as np
import base64

async def send_audio():
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Generate test audio
        audio = np.random.randn(16000).astype(np.float32)
        audio_bytes = audio.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        # Send audio
        await websocket.send(json.dumps({
            'type': 'audio',
            'audio': audio_base64,
            'sample_rate': 16000,
            'timestamp': 0.0
        }))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(data)

asyncio.run(send_audio())
```

## API Reference

### Pipeline Methods

```python
# Initialize pipeline
pipeline = MultiSpeakerASRPipeline(config_path="config/config.yaml")

# Process file
segments = pipeline.process_file("audio.wav")

# Process audio stream (async)
async for result in pipeline.process_audio_stream(audio_generator):
    # Handle results

# Get performance metrics
metrics = pipeline.get_performance_metrics()

# Save/load speaker profiles
pipeline.save_speaker_profiles("profiles.pkl")
pipeline.load_speaker_profiles("profiles.pkl")
```

### Corrector Methods

```python
# Initialize correctors
nlp_corrector = NLPErrorCorrector()
realtime_corrector = RealTimeErrorCorrector(max_latency_ms=100)
speaker_corrector = SpeakerAwareCorrector()

# Detect errors
errors = nlp_corrector.detect_errors("garbled text###")

# Correct segment
corrected = nlp_corrector.correct_segment(segment)

# Real-time correction
result = realtime_corrector.correct_with_deadline(text, context)

# Speaker-specific
corrected = speaker_corrector.speaker_specific_correction(text, speaker_id)
```

## Configuration

### Key Configuration Options

```yaml
# config/config.yaml

# NLP Correction
nlp_correction:
    primary_model: "t5-base" # Model size: t5-small, t5-base, t5-large
    max_latency_ms: 100 # Real-time constraint
    context_window_size: 5 # Context history size
    correction_threshold: 0.6 # Min confidence to apply correction

    strategies:
        pattern_based: true # Fast rule-based fixes
        masked_lm: true # BERT-based correction
        seq2seq: true # T5-based correction
        phonetic: true # Sound-alike corrections
        contextual: true # Context-aware correction
        speaker_specific: true # Learn from speakers

# Audio Processing
audio:
    sample_rate: 16000
    chunk_duration: 0.5 # Streaming chunk size

# ASR
asr:
    model_type: "whisper" # or "faster
```
