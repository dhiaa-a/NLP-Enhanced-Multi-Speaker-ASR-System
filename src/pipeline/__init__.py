# src/pipeline/__init__.py
"""
Pipeline Orchestration Module
Manages the complete audio processing pipeline.
"""

from .orchestrator import (
    MultiSpeakerASRPipeline,
    AudioSegment,
    ProcessingResult
)

__all__ = [
    "MultiSpeakerASRPipeline",
    "AudioSegment",
    "ProcessingResult"
]