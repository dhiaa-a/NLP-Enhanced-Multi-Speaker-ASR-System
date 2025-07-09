# API Reference

Complete API documentation for the Multi-Speaker ASR with NLP Enhancement system.

## Table of Contents

1. [Pipeline Classes](#pipeline-classes)
2. [NLP Correction Classes](#nlp-correction-classes)
3. [Audio Processing Classes](#audio-processing-classes)
4. [Data Classes](#data-classes)
5. [Utilities](#utilities)

## Pipeline Classes

### MultiSpeakerASRPipeline

Main pipeline orchestrator for processing audio through all stages.

```python
class MultiSpeakerASRPipeline(config_path: str = "config/config.yaml")
```

#### Parameters

-   `config_path` (str): Path to configuration YAML file

#### Methods

##### process_file

```python
def process_file(audio_file_path: str) -> List[TranscriptionSegment]
```

Process a single audio file synchronously.

**Parameters:**

-   `audio_file_path` (str): Path to audio file (WAV, MP3, etc.)

**Returns:**

-   List[TranscriptionSegment]: Processed segments with corrections

**Example:**

```python
pipeline = MultiSpeakerASRPipeline()
segments = pipeline.process_file("meeting.wav")
```

##### process_audio_stream

```python
async def process_audio_stream(
    audio_stream: AsyncGenerator[AudioSegment, None]
) -> AsyncGenerator[ProcessingResult, None]
```

Process streaming audio asynchronously.

**Parameters:**

-   `audio_stream`: Async generator yielding AudioSegment objects

**Yields:**

-   ProcessingResult: Results for each processed chunk

**Example:**

```python
async for result in pipeline.process_audio_stream(audio_generator):
    print(f"Latency: {result.total_latency_ms}ms")
```

##### get_performance_metrics

```python
def get_performance_metrics() -> Dict
```

Get current performance statistics.

**Returns:**

-   Dict containing:
    -   `processed_segments`: Total segments processed
    -   `average_latency_ms`: Average processing latency
    -   `stage_latencies`: Latency breakdown by stage
    -   `nlp_correction_stats`: NLP correction statistics

##### save_speaker_profiles / load_speaker_profiles

```python
def save_speaker_profiles(filepath: str) -> None
def load_speaker_profiles(filepath: str) -> None
```

Save or load learned speaker profiles.

## NLP Correction Classes

### NLPErrorCorrector

Core error correction class using context-aware language models.

```python
class NLPErrorCorrector(config_path: str = "config/config.yaml")
```

#### Methods

##### detect_errors

```python
def detect_errors(text: str) -> Dict[str, float]
```

Analyze text for ASR errors.

**Parameters:**

-   `text` (str): Raw ASR output

**Returns:**

-   Dict with error scores:
    -   `incomplete_words`: Score for word fragments
    -   `overlap_artifacts`: Score for overlapping speech artifacts
    -   `grammar_issues`: Grammar error score
    -   `confidence`: Overall confidence (0-1)

**Example:**

```python
errors = corrector.detect_errors("The meet- is at thre###")
if errors['confidence'] < 0.5:
    print("High error rate detected")
```

##### correct_segment

```python
def correct_segment(segment: TranscriptionSegment) -> TranscriptionSegment
```

Apply correction to a transcription segment.

**Parameters:**

-   `segment`: TranscriptionSegment with raw text

**Returns:**

-   TranscriptionSegment with corrected_text filled

##### get_context_window

```python
def get_context_window() -> List[str]
```

Get current conversation context.

**Returns:**

-   List of recent utterances for context

### RealTimeErrorCorrector

Optimized corrector for low-latency applications.

```python
class RealTimeErrorCorrector(
    max_latency_ms: int = 100,
    config: dict = None
)
```

#### Parameters

-   `max_latency_ms` (int): Maximum allowed latency
-   `config` (dict): Optional configuration overrides

#### Methods

##### correct_with_deadline

```python
def correct_with_deadline(
    text: str,
    context: Optional[List[str]] = None
) -> CorrectionResult
```

Correct text within latency deadline.

**Parameters:**

-   `text` (str): Text to correct
-   `context` (List[str]): Optional context

**Returns:**

-   CorrectionResult with:
    -   `text`: Corrected text
    -   `latency_ms`: Processing time
    -   `strategy`: Strategy used
    -   `confidence`: Result confidence

**Example:**

```python
result = corrector.correct_with_deadline("meet- tomorrow", deadline_ms=50)
print(f"Corrected in {result.latency_ms}ms: {result.text}")
```

##### batch_correct

```python
def batch_correct(
    texts: List[str],
    context: Optional[List[str]] = None
) -> List[CorrectionResult]
```

Process multiple texts efficiently.

### SpeakerAwareCorrector

Maintains speaker-specific language models.

```python
class SpeakerAwareCorrector(max_history_size: int = 100)
```

#### Methods

##### update_speaker_model

```python
def update_speaker_model(
    speaker_id: int,
    corrected_text: str,
    raw_text: Optional[str] = None
) -> None
```

Update speaker's language model with new utterance.

**Parameters:**

-   `speaker_id` (int): Speaker identifier
-   `corrected_text` (str): Corrected utterance
-   `raw_text` (str): Optional raw text for learning errors

##### speaker_specific_correction

```python
def speaker_specific_correction(
    text: str,
    speaker_id: int,
    context: List[str] = None
) -> str
```

Apply speaker-specific correction.

**Returns:**

-   Corrected text using speaker's patterns

##### get_speaker_statistics

```python
def get_speaker_statistics(speaker_id: int) -> Dict
```

Get statistics for a speaker.

**Returns:**

-   Dict with:
    -   `vocabulary_size`: Unique words
    -   `average_utterance_length`: Avg words per utterance
    -   `speaking_style`: Style metrics
    -   `common_phrases`: Frequent phrases

### CorrectionStrategies

Individual correction strategies for different error types.

```python
class CorrectionStrategies(device: torch.device = None)
```

#### Methods

##### pattern_based_correction

```python
def pattern_based_correction(text: str) -> str
```

Fast rule-based correction for common patterns.

##### masked_language_model_correction

```python
def masked_language_model_correction(text: str) -> str
```

Use BERT to predict corrupted parts.

##### phonetic_correction

```python
def phonetic_correction(text: str) -> str
```

Fix sound-alike word confusions.

##### contextual_correction

```python
def contextual_correction(
    text: str,
    context: List[str]
) -> str
```

Use conversation context for correction.

## Audio Processing Classes

### NoiseReducer

Audio noise reduction using denoising autoencoders.

```python
class NoiseReducer(config: Dict)
```

#### Methods

##### process

```python
def process(audio: np.ndarray) -> np.ndarray
```

Remove noise from audio.

**Parameters:**

-   `audio` (np.ndarray): Input audio array

**Returns:**

-   np.ndarray: Denoised audio

### SpeakerSeparator

Separate multiple speakers from mixed audio.

```python
class SpeakerSeparator(config: Dict)
```

#### Methods

##### separate

```python
def separate(audio: np.ndarray) -> List[Dict]
```

Separate speakers from audio.

**Returns:**

-   List of dicts with:
    -   `speaker_id`: Speaker identifier
    -   `audio`: Separated audio
    -   `confidence`: Separation confidence

### ASRInterface

Unified interface for ASR engines.

```python
class ASRInterface(config: Dict)
```

#### Methods

##### transcribe

```python
def transcribe(audio: np.ndarray) -> str
```

Convert audio to text.

**Parameters:**

-   `audio` (np.ndarray): Audio array (16kHz mono)

**Returns:**

-   str: Transcribed text

## Data Classes

### TranscriptionSegment

```python
@dataclass
class TranscriptionSegment:
    speaker_id: int
    start_time: float
    end_time: float
    raw_text: str
    corrected_text: Optional[str] = None
    confidence: float = 1.0
    context_window: Optional[List[str]] = None
    correction_metadata: Optional[Dict] = None
```

Represents a transcribed segment with metadata.

### AudioSegment

```python
@dataclass
class AudioSegment:
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float
```

Audio data with metadata for processing.

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    segments: List[TranscriptionSegment]
    total_latency_ms: float
    stage_latencies: Dict[str, float]
    audio_timestamp: float
```

Complete result from pipeline processing.

### CorrectionResult

```python
@dataclass
class CorrectionResult:
    text: str
    latency_ms: float
    strategy: str
    confidence: float
```

Result from real-time correction.

### SpeakerProfile

```python
@dataclass
class SpeakerProfile:
    speaker_id: int
    vocabulary: Set[str]
    phrase_patterns: List[str]
    word_frequencies: Counter
    speaking_style: Dict[str, float]
    utterance_history: List[str]
    # ... more fields
```

Complete profile for a speaker.

## Utilities

### Configuration Loading

```python
def load_config(config_path: str) -> dict
```

Load YAML configuration file.

### Performance Monitoring

```python
def get_performance_stats() -> Dict
```

Get system performance statistics.

### Model Management

```python
def download_models() -> None
def verify_models() -> bool
```

Download and verify required models.

## WebSocket API

### Server Messages

#### Transcription Result

```json
{
	"type": "transcription",
	"timestamp": 1234567890.123,
	"processing_time_ms": 45.2,
	"segments": [
		{
			"speaker_id": 1,
			"start_time": 0.0,
			"end_time": 3.5,
			"raw_text": "The meet- is at thre###",
			"corrected_text": "The meeting is at three",
			"confidence": 0.85
		}
	]
}
```

#### Error Message

```json
{
	"type": "error",
	"message": "Processing failed: Invalid audio format"
}
```

### Client Messages

#### Audio Data

```json
{
	"type": "audio",
	"audio": "base64_encoded_audio_data",
	"sample_rate": 16000,
	"timestamp": 1234567890.123
}
```

#### Configuration Update

```json
{
	"type": "config",
	"config": {
		"language": "en",
		"max_speakers": 4
	}
}
```

## Error Handling

All methods may raise:

-   `ValueError`: Invalid input parameters
-   `RuntimeError`: Processing errors
-   `TimeoutError`: Deadline exceeded (real-time methods)

Example error handling:

```python
try:
    segments = pipeline.process_file("audio.wav")
except ValueError as e:
    logger.error(f"Invalid input: {e}")
except RuntimeError as e:
    logger.error(f"Processing failed: {e}")
    # Use fallback
```

## Performance Considerations

### Memory Usage

-   Base models: ~2GB GPU memory
-   Large models: ~8GB GPU memory
-   CPU processing: ~4GB RAM minimum

### Latency Guidelines

-   Ultra-fast: <20ms (pattern-based only)
-   Fast: <50ms (small models)
-   Standard: <100ms (base models)
-   High-quality: <500ms (large models)

### Optimization Tips

1. Use GPU when available
2. Enable caching for repeated content
3. Adjust batch size based on memory
4. Use appropriate model size for use case
