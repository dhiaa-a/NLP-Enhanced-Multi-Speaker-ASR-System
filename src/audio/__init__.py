# src/audio/__init__.py
"""
Audio Processing Module
Handles noise reduction and speaker separation.
"""

from .noise_reduction import NoiseReducer, create_noise_reducer
from .speaker_separation import SpeakerSeparator, create_speaker_separator

__all__ = [
    "NoiseReducer",
    "create_noise_reducer",
    "SpeakerSeparator",
    "create_speaker_separator"
]