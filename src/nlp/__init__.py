# src/nlp/__init__.py
"""
NLP Error Correction Module
Core innovation for fixing ASR errors in overlapping speech.
"""

from .error_corrector import NLPErrorCorrector, TranscriptionSegment
from .correction_strategies import CorrectionStrategies
from .speaker_models import SpeakerAwareCorrector, SpeakerProfile
from .realtime_corrector import RealTimeErrorCorrector, CorrectionResult

__all__ = [
    "NLPErrorCorrector",
    "TranscriptionSegment",
    "CorrectionStrategies",
    "SpeakerAwareCorrector",
    "SpeakerProfile",
    "RealTimeErrorCorrector",
    "CorrectionResult"
]