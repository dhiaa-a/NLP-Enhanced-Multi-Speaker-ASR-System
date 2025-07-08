# src/asr/__init__.py
"""
ASR Interface Module
Provides unified interface to various ASR engines.
"""

from .asr_interface import ASRInterface, create_asr_interface

__all__ = [
    "ASRInterface",
    "create_asr_interface"
]