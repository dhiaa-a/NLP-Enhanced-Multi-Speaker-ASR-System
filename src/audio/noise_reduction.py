"""
Noise Reduction Module
Implements denoising autoencoder for audio cleaning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import torchaudio
from loguru import logger

class DenoisingAutoencoder(nn.Module):
    """Simple denoising autoencoder for audio noise reduction"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class NoiseReducer:
    """
    Noise reduction interface for the pipeline.
    In production, this would use a pre-trained denoising model.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config.get('enabled', True):
            self._init_model()
        else:
            self.model = None
            logger.info("Noise reduction disabled")
    
    def _init_model(self):
        """Initialize the denoising model"""
        try:
            # In production, load a pre-trained model
            model_path = self.config.get('model_path')
            
            if model_path and os.path.exists(model_path):
                # Load pre-trained model
                self.model = torch.load(model_path, map_location=self.device)
                logger.info(f"Loaded noise reduction model from {model_path}")
            else:
                # Use simple denoising autoencoder
                input_dim = self.config.get('input_dim', 128)
                hidden_dim = self.config.get('hidden_dim', 64)
                self.model = DenoisingAutoencoder(input_dim, hidden_dim).to(self.device)
                self.model.eval()
                logger.info("Using default denoising autoencoder")
                
        except Exception as e:
            logger.error(f"Failed to initialize noise reduction model: {e}")
            self.model = None
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio to reduce noise.
        
        Args:
            audio: Input audio array
            
        Returns:
            Denoised audio array
        """
        if self.model is None or not self.config.get('enabled', True):
            return audio
        
        try:
            # Convert to spectrogram for processing
            audio_tensor = torch.from_numpy(audio).float()
            
            # Simple spectral gating for demonstration
            # In production, use the trained model
            stft = torch.stft(
                audio_tensor,
                n_fft=512,
                hop_length=256,
                window=torch.hann_window(512),
                return_complex=True
            )
            
            # Apply simple noise reduction
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Noise gate threshold
            threshold = magnitude.mean() * 0.1
            magnitude = torch.where(magnitude > threshold, magnitude, torch.zeros_like(magnitude))
            
            # Reconstruct
            stft_denoised = magnitude * torch.exp(1j * phase)
            audio_denoised = torch.istft(
                stft_denoised,
                n_fft=512,
                hop_length=256,
                window=torch.hann_window(512)
            )
            
            # Ensure same length
            if len(audio_denoised) > len(audio):
                audio_denoised = audio_denoised[:len(audio)]
            elif len(audio_denoised) < len(audio):
                padding = torch.zeros(len(audio) - len(audio_denoised))
                audio_denoised = torch.cat([audio_denoised, padding])
            
            return audio_denoised.numpy()
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return audio
    
    def process_batch(self, audio_batch: np.ndarray) -> np.ndarray:
        """Process a batch of audio samples"""
        return np.array([self.process(audio) for audio in audio_batch])

# Placeholder for integration with existing noise reduction libraries
def create_noise_reducer(config: Dict) -> NoiseReducer:
    """Factory function to create appropriate noise reducer"""
    
    model_type = config.get('model_type', 'denoising_autoencoder')
    
    if model_type == 'denoising_autoencoder':
        return NoiseReducer(config)
    else:
        # Add support for other models here
        logger.warning(f"Unknown model type: {model_type}, using default")
        return NoiseReducer(config)

import os