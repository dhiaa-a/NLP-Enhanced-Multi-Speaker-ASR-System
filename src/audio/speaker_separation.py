"""
Speaker Separation Module
Implements speaker separation using deep clustering or pre-trained models.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
from loguru import logger

try:
    from speechbrain.pretrained import SepformerSeparation
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("SpeechBrain not available. Using mock separator.")

class SpeakerSeparator:
    """
    Speaker separation interface for the pipeline.
    Uses SpeechBrain's SepFormer or falls back to mock separation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config.get('enabled', True):
            self._init_model()
        else:
            self.model = None
            logger.info("Speaker separation disabled")
    
    def _init_model(self):
        """Initialize the speaker separation model"""
        model_type = self.config.get('model_type', 'speechbrain')
        
        if model_type == 'speechbrain' and SPEECHBRAIN_AVAILABLE:
            try:
                model_name = self.config.get('model_name', 'speechbrain/sepformer-wsj02mix')
                self.model = SepformerSeparation.from_hparams(
                    source=model_name,
                    savedir="models/sepformer",
                    run_opts={"device": str(self.device)}
                )
                logger.info(f"Loaded SpeechBrain separator: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load SpeechBrain model: {e}")
                self.model = None
        else:
            # Use mock separator for demonstration
            self.model = MockSeparator(self.config)
            logger.info("Using mock speaker separator")
    
    def separate(self, audio: np.ndarray) -> List[Dict]:
        """
        Separate speakers from mixed audio.
        
        Args:
            audio: Mixed audio array (mono, 16kHz expected)
            
        Returns:
            List of dictionaries containing separated audio and metadata
        """
        if self.model is None or not self.config.get('enabled', True):
            # Return original audio as single speaker
            return [{
                'speaker_id': 0,
                'audio': audio,
                'start_time': 0.0,
                'confidence': 1.0
            }]
        
        try:
            if hasattr(self.model, 'separate_batch'):
                # SpeechBrain interface
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                
                # Separate speakers
                separated = self.model.separate_batch(audio_tensor)
                
                # Convert back to numpy
                results = []
                for i, source in enumerate(separated[0]):
                    results.append({
                        'speaker_id': i,
                        'audio': source.cpu().numpy(),
                        'start_time': 0.0,
                        'confidence': 0.9
                    })
                
                return results
            else:
                # Mock separator
                return self.model.separate(audio)
                
        except Exception as e:
            logger.error(f"Speaker separation failed: {e}")
            # Fallback: return original audio
            return [{
                'speaker_id': 0,
                'audio': audio,
                'start_time': 0.0,
                'confidence': 0.5
            }]
    
    def separate_with_vad(self, audio: np.ndarray) -> List[Dict]:
        """
        Separate speakers with Voice Activity Detection.
        Returns segments with timing information.
        """
        # First separate speakers
        separated = self.separate(audio)
        
        # Apply VAD to each separated source
        results = []
        for source_data in separated:
            segments = self._apply_vad(source_data['audio'])
            
            for segment in segments:
                results.append({
                    'speaker_id': source_data['speaker_id'],
                    'audio': segment['audio'],
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'confidence': source_data['confidence']
                })
        
        return results
    
    def _apply_vad(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """
        Simple Voice Activity Detection.
        In production, use a proper VAD model.
        """
        # Simple energy-based VAD
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        hop_size = int(0.01 * sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy.append(np.mean(frame ** 2))
        
        energy = np.array(energy)
        threshold = np.mean(energy) * 0.5
        
        # Find voice segments
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, e in enumerate(energy):
            if e > threshold and not in_segment:
                start_idx = i
                in_segment = True
            elif e <= threshold and in_segment:
                # End of segment
                start_time = start_idx * hop_size / sample_rate
                end_time = i * hop_size / sample_rate
                
                if end_time - start_time > 0.1:  # Minimum 100ms
                    start_sample = start_idx * hop_size
                    end_sample = min(i * hop_size, len(audio))
                    
                    segments.append({
                        'audio': audio[start_sample:end_sample],
                        'start_time': start_time,
                        'end_time': end_time
                    })
                
                in_segment = False
        
        # Handle last segment
        if in_segment:
            start_time = start_idx * hop_size / sample_rate
            end_time = len(audio) / sample_rate
            start_sample = start_idx * hop_size
            
            segments.append({
                'audio': audio[start_sample:],
                'start_time': start_time,
                'end_time': end_time
            })
        
        return segments if segments else [{
            'audio': audio,
            'start_time': 0.0,
            'end_time': len(audio) / sample_rate
        }]

class MockSeparator:
    """Mock separator for testing without SpeechBrain"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_speakers = config.get('max_speakers', 4)
    
    def separate(self, audio: np.ndarray) -> List[Dict]:
        """
        Mock separation - splits audio into overlapping segments.
        Simulates multiple speakers for testing.
        """
        duration = len(audio) / 16000  # Assuming 16kHz
        
        if duration < 1.0:
            # Too short to separate
            return [{
                'speaker_id': 0,
                'audio': audio,
                'start_time': 0.0,
                'confidence': 0.8
            }]
        
        # Create 2-3 mock speakers with overlapping segments
        num_speakers = min(3, self.max_speakers)
        results = []
        
        for i in range(num_speakers):
            # Create slightly different version of audio
            # In reality, this would be actual separated audio
            speaker_audio = audio.copy()
            
            # Apply simple frequency filter to differentiate
            if i == 1:
                # High-pass filter effect
                speaker_audio = speaker_audio * 0.7
            elif i == 2:
                # Low-pass filter effect
                speaker_audio = speaker_audio * 0.5
            
            # Add some noise to simulate separation artifacts
            noise = np.random.normal(0, 0.01, len(speaker_audio))
            speaker_audio = speaker_audio + noise
            
            results.append({
                'speaker_id': i,
                'audio': speaker_audio,
                'start_time': 0.0,
                'confidence': 0.7 - (i * 0.1)
            })
        
        return results

def create_speaker_separator(config: Dict) -> SpeakerSeparator:
    """Factory function to create appropriate speaker separator"""
    return SpeakerSeparator(config)