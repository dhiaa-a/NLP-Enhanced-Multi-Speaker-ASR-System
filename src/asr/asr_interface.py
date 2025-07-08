"""
ASR Interface Module
Provides interface to various ASR engines (Whisper, etc.)
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from loguru import logger
import torch

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("OpenAI Whisper not available. Using mock ASR.")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

class ASRInterface:
    """
    Unified interface for different ASR engines.
    Supports Whisper, Faster-Whisper, and mock ASR for testing.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config.get('model_type', 'whisper')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the ASR model based on configuration"""
        if self.model_type == 'faster-whisper' and FASTER_WHISPER_AVAILABLE:
            self._init_faster_whisper()
        elif self.model_type == 'whisper' and WHISPER_AVAILABLE:
            self._init_whisper()
        else:
            self._init_mock_asr()
    
    def _init_whisper(self):
        """Initialize OpenAI Whisper"""
        try:
            model_size = self.config.get('model_size', 'base')
            self.model = whisper.load_model(model_size, device=self.device)
            self.model_type = 'whisper'
            logger.info(f"Loaded Whisper model: {model_size}")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self._init_mock_asr()
    
    def _init_faster_whisper(self):
        """Initialize Faster-Whisper (optimized version)"""
        try:
            model_size = self.config.get('model_size', 'base')
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type
            )
            self.model_type = 'faster-whisper'
            logger.info(f"Loaded Faster-Whisper model: {model_size}")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper: {e}")
            self._init_whisper()  # Fall back to regular Whisper
    
    def _init_mock_asr(self):
        """Initialize mock ASR for testing"""
        self.model = MockASR(self.config)
        self.model_type = 'mock'
        logger.info("Using mock ASR for testing")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (mono, 16kHz expected)
            
        Returns:
            Transcribed text
        """
        if self.model_type == 'whisper':
            return self._transcribe_whisper(audio)
        elif self.model_type == 'faster-whisper':
            return self._transcribe_faster_whisper(audio)
        else:
            return self._transcribe_mock(audio)
    
    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        """Transcribe using OpenAI Whisper"""
        try:
            # Whisper expects float32 audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                language=self.config.get('language', 'en'),
                task=self.config.get('task', 'transcribe'),
                temperature=self.config.get('temperature', 0.0),
                beam_size=self.config.get('beam_size', 5)
            )
            
            return result['text'].strip()
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return "[Transcription failed]"
    
    def _transcribe_faster_whisper(self, audio: np.ndarray) -> str:
        """Transcribe using Faster-Whisper"""
        try:
            # Faster-Whisper expects float32 audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language=self.config.get('language', 'en'),
                task=self.config.get('task', 'transcribe'),
                beam_size=self.config.get('beam_size', 5),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect text from segments
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        except Exception as e:
            logger.error(f"Faster-Whisper transcription failed: {e}")
            return "[Transcription failed]"
    
    def _transcribe_mock(self, audio: np.ndarray) -> str:
        """Mock transcription for testing"""
        return self.model.transcribe(audio)
    
    def transcribe_with_timestamps(self, audio: np.ndarray) -> List[Dict]:
        """
        Transcribe audio with word-level timestamps.
        
        Returns:
            List of dictionaries with 'text', 'start', and 'end' keys
        """
        if self.model_type == 'whisper':
            try:
                result = self.model.transcribe(
                    audio,
                    language=self.config.get('language', 'en'),
                    task=self.config.get('task', 'transcribe'),
                    word_timestamps=True
                )
                
                segments = []
                for segment in result['segments']:
                    for word in segment.get('words', []):
                        segments.append({
                            'text': word['word'],
                            'start': word['start'],
                            'end': word['end']
                        })
                
                return segments
            except:
                pass
        
        # Fallback: return single segment
        return [{
            'text': self.transcribe(audio),
            'start': 0.0,
            'end': len(audio) / 16000
        }]

class MockASR:
    """Mock ASR for testing without real models"""
    
    def __init__(self, config: Dict):
        self.config = config
        # Simulated ASR errors for testing
        self.error_patterns = [
            "I think we sh## meet tomor- at thre-",
            "Yeah I agr## with tha+ but we should also consid## the budg##",
            "The proj### deadline is next fri###",
            "Can you he## me? The connec### is really b##",
            "Let's disc### the new feat### in the next spri##",
            "I'll send the rep### by end of d##"
        ]
        self.pattern_index = 0
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Return mock transcription with typical ASR errors.
        Cycles through predefined error patterns for testing.
        """
        # Use audio length to determine which pattern to return
        # This ensures consistency for testing
        duration = len(audio) / 16000  # Assuming 16kHz
        
        if duration < 0.5:
            return "Too short"
        
        # Cycle through patterns
        pattern = self.error_patterns[self.pattern_index % len(self.error_patterns)]
        self.pattern_index += 1
        
        # Add some variability based on audio characteristics
        if np.std(audio) < 0.01:
            # Very quiet audio
            return pattern.replace("#", "###")
        elif np.std(audio) > 0.5:
            # Very loud audio
            return pattern.upper()
        
        return pattern

def create_asr_interface(config: Dict) -> ASRInterface:
    """Factory function to create ASR interface"""
    return ASRInterface(config)