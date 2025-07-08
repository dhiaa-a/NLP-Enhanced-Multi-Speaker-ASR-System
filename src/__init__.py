# src/__init__.py
"""
Multi-Speaker ASR with NLP Enhancement
A novel approach to handling overlapping speech in real-time.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Make key components easily importable
from .pipeline.orchestrator import MultiSpeakerASRPipeline
from .nlp.error_corrector import NLPErrorCorrector
from .nlp.realtime_corrector import RealTimeErrorCorrector

__all__ = [
    "MultiSpeakerASRPipeline",
    "NLPErrorCorrector", 
    "RealTimeErrorCorrector"
]