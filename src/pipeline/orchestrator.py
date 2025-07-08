"""
Pipeline Orchestrator Module
Manages the complete audio processing pipeline from input to corrected output.
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
import yaml
import time
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import torch
import queue
import threading

# Import our modules
from ..nlp.error_corrector import NLPErrorCorrector, TranscriptionSegment
from ..nlp.speaker_models import SpeakerAwareCorrector
from ..nlp.realtime_corrector import RealTimeErrorCorrector, CorrectionResult
from ..audio.noise_reduction import NoiseReducer
from ..audio.speaker_separation import SpeakerSeparator
from ..asr.asr_interface import ASRInterface

@dataclass
class AudioSegment:
    """Represents an audio segment for processing"""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float

@dataclass
class ProcessingResult:
    """Complete result from processing pipeline"""
    segments: List[TranscriptionSegment]
    total_latency_ms: float
    stage_latencies: Dict[str, float]
    audio_timestamp: float

class MultiSpeakerASRPipeline:
    """
    Main pipeline orchestrator for multi-speaker ASR with NLP correction.
    Coordinates all processing stages and manages real-time constraints.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the pipeline with configuration"""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_components()
        
        # Processing state
        self.is_running = False
        self.processing_queue = queue.Queue(maxsize=self.config['pipeline']['buffer_size'])
        self.result_queue = queue.Queue()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['pipeline']['max_workers']
        )
        
        # Performance monitoring
        self.performance_metrics = {
            'processed_segments': 0,
            'average_latency_ms': 0,
            'stage_latencies': {},
            'error_count': 0
        }
        
        logger.info("Multi-speaker ASR pipeline initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _init_components(self):
        """Initialize all pipeline components"""
        try:
            # Audio processing components
            self.noise_reducer = NoiseReducer(self.config['noise_reduction'])
            self.speaker_separator = SpeakerSeparator(self.config['speaker_separation'])
            
            # ASR component
            self.asr = ASRInterface(self.config['asr'])
            
            # NLP correction components
            self.nlp_corrector = NLPErrorCorrector(self.config)
            self.speaker_corrector = SpeakerAwareCorrector()
            self.realtime_corrector = RealTimeErrorCorrector(
                max_latency_ms=self.config['nlp_correction']['max_latency_ms'],
                config=self.config['nlp_correction']
            )
            
            # Warmup models
            if self.config.get('optimization', {}).get('warmup', True):
                self._warmup_models()
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _warmup_models(self):
        """Warmup models for better initial performance"""
        logger.info("Warming up models...")
        
        # Create dummy audio
        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second
        
        try:
            # Run through pipeline
            self.noise_reducer.process(dummy_audio)
            self.realtime_corrector.warmup()
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    async def process_audio_stream(self, audio_stream: AsyncGenerator[AudioSegment, None]) -> AsyncGenerator[ProcessingResult, None]:
        """
        Main processing method for audio stream.
        
        Args:
            audio_stream: Async generator of audio segments
            
        Yields:
            ProcessingResult for each processed segment
        """
        self.is_running = True
        
        try:
            async for audio_segment in audio_stream:
                if not self.is_running:
                    break
                    
                # Process segment
                result = await self.process_audio_segment(audio_segment)
                
                # Update metrics
                self._update_metrics(result)
                
                yield result
                
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            self.performance_metrics['error_count'] += 1
        finally:
            self.is_running = False
    
    async def process_audio_segment(self, audio_segment: AudioSegment) -> ProcessingResult:
        """
        Process a single audio segment through the pipeline.
        
        Args:
            audio_segment: Audio segment to process
            
        Returns:
            ProcessingResult with transcriptions and metrics
        """
        start_time = time.time()
        stage_latencies = {}
        
        try:
            # Stage 1: Noise Reduction
            stage_start = time.time()
            clean_audio = await self._async_noise_reduction(audio_segment.data)
            stage_latencies['noise_reduction'] = (time.time() - stage_start) * 1000
            
            # Stage 2: Speaker Separation
            stage_start = time.time()
            speaker_segments = await self._async_speaker_separation(clean_audio)
            stage_latencies['speaker_separation'] = (time.time() - stage_start) * 1000
            
            # Stage 3 & 4: ASR + NLP Correction (in parallel for each speaker)
            stage_start = time.time()
            transcription_segments = await self._process_speakers_parallel(
                speaker_segments, 
                audio_segment.timestamp
            )
            stage_latencies['asr_nlp_correction'] = (time.time() - stage_start) * 1000
            
            # Total latency
            total_latency_ms = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                segments=transcription_segments,
                total_latency_ms=total_latency_ms,
                stage_latencies=stage_latencies,
                audio_timestamp=audio_segment.timestamp
            )
            
        except Exception as e:
            logger.error(f"Segment processing failed: {e}")
            # Return empty result on error
            return ProcessingResult(
                segments=[],
                total_latency_ms=(time.time() - start_time) * 1000,
                stage_latencies=stage_latencies,
                audio_timestamp=audio_segment.timestamp
            )
    
    async def _async_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Async wrapper for noise reduction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.noise_reducer.process,
            audio
        )
    
    async def _async_speaker_separation(self, audio: np.ndarray) -> List[Dict]:
        """Async wrapper for speaker separation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.speaker_separator.separate,
            audio
        )
    
    async def _process_speakers_parallel(self, speaker_segments: List[Dict], timestamp: float) -> List[TranscriptionSegment]:
        """
        Process multiple speakers in parallel.
        
        Args:
            speaker_segments: List of separated speaker audio
            timestamp: Original audio timestamp
            
        Returns:
            List of transcription segments with corrections
        """
        tasks = []
        
        for speaker_data in speaker_segments:
            task = self._process_single_speaker(
                speaker_data['audio'],
                speaker_data['speaker_id'],
                timestamp + speaker_data.get('start_time', 0)
            )
            tasks.append(task)
        
        # Process all speakers in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any failed results
        valid_results = []
        for result in results:
            if isinstance(result, TranscriptionSegment):
                valid_results.append(result)
            else:
                logger.warning(f"Speaker processing failed: {result}")
        
        return valid_results
    
    async def _process_single_speaker(self, audio: np.ndarray, speaker_id: int, start_time: float) -> TranscriptionSegment:
        """
        Process audio from a single speaker.
        
        Args:
            audio: Speaker's audio data
            speaker_id: Speaker identifier
            start_time: Start timestamp
            
        Returns:
            TranscriptionSegment with corrected text
        """
        # ASR transcription
        loop = asyncio.get_event_loop()
        raw_text = await loop.run_in_executor(
            self.executor,
            self.asr.transcribe,
            audio
        )
        
        # Create segment
        segment = TranscriptionSegment(
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=start_time + len(audio) / 16000,  # Assuming 16kHz
            raw_text=raw_text,
            context_window=self.nlp_corrector.get_context_window()
        )
        
        # Apply NLP correction based on quality and time constraints
        if self.config['nlp_correction']['strategies']['realtime']:
            # Use real-time corrector
            correction_result = await self.realtime_corrector.correct_async(
                raw_text,
                segment.context_window
            )
            segment.corrected_text = correction_result.text
            segment.confidence = correction_result.confidence
        else:
            # Use standard corrector
            segment = self.nlp_corrector.correct_segment(segment)
        
        # Apply speaker-specific corrections if enabled
        if self.config['nlp_correction']['strategies']['speaker_specific']:
            segment.corrected_text = self.speaker_corrector.speaker_specific_correction(
                segment.corrected_text or segment.raw_text,
                speaker_id,
                segment.context_window
            )
            
            # Update speaker model
            self.speaker_corrector.update_speaker_model(
                speaker_id,
                segment.corrected_text,
                segment.raw_text
            )
        
        return segment
    
    def _update_metrics(self, result: ProcessingResult):
        """Update performance metrics"""
        self.performance_metrics['processed_segments'] += len(result.segments)
        
        # Update average latency
        n = self.performance_metrics['processed_segments']
        current_avg = self.performance_metrics['average_latency_ms']
        self.performance_metrics['average_latency_ms'] = (
            (current_avg * (n - 1) + result.total_latency_ms) / n
        )
        
        # Update stage latencies
        for stage, latency in result.stage_latencies.items():
            if stage not in self.performance_metrics['stage_latencies']:
                self.performance_metrics['stage_latencies'][stage] = latency
            else:
                # Running average
                current = self.performance_metrics['stage_latencies'][stage]
                self.performance_metrics['stage_latencies'][stage] = (
                    (current * (n - 1) + latency) / n
                )
    
    def process_file(self, audio_file_path: str) -> List[TranscriptionSegment]:
        """
        Process an audio file (synchronous interface).
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            List of transcription segments
        """
        # Load audio file
        import librosa
        audio, sr = librosa.load(audio_file_path, sr=16000)
        
        # Create audio segment
        segment = AudioSegment(
            data=audio,
            sample_rate=sr,
            timestamp=0.0,
            duration=len(audio) / sr
        )
        
        # Process synchronously
        async def process():
            result = await self.process_audio_segment(segment)
            return result.segments
        
        return asyncio.run(process())
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Add component-specific metrics
        metrics['nlp_correction_stats'] = self.realtime_corrector.get_performance_stats()
        
        return metrics
    
    def save_speaker_profiles(self, filepath: str):
        """Save learned speaker profiles"""
        self.speaker_corrector.save_profiles(filepath)
    
    def load_speaker_profiles(self, filepath: str):
        """Load speaker profiles"""
        self.speaker_corrector.load_profiles(filepath)
    
    def stop(self):
        """Stop the pipeline"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Pipeline stopped")