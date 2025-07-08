"""
Main NLP Error Correction Module for Multi-Speaker ASR
This module contains the base error correction class with core functionality.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import re
import yaml
from loguru import logger

@dataclass
class TranscriptionSegment:
    """Represents a transcribed segment with metadata"""
    speaker_id: int
    start_time: float
    end_time: float
    raw_text: str
    corrected_text: Optional[str] = None
    confidence: float = 1.0
    context_window: Optional[List[str]] = None
    correction_metadata: Optional[Dict] = None

class NLPErrorCorrector:
    """
    Core NLP Error Correction class for fixing garbled ASR output
    using context-aware language models and multiple correction strategies.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the NLP Error Corrector with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        
        # Initialize models
        self._init_models()
        
        # Initialize correction components
        self.context_buffer = []
        self.correction_cache = {}
        
        logger.info("NLP Error Corrector initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['nlp_correction']
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'primary_model': 't5-base',
                'context_window_size': 5,
                'max_latency_ms': 100,
                'correction_threshold': 0.6
            }
    
    def _setup_device(self) -> torch.device:
        """Setup computation device (GPU/CPU)"""
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def _init_models(self):
        """Initialize all required models"""
        try:
            # Primary correction model (T5 or BART)
            model_name = self.config.get('primary_model', 't5-base')
            logger.info(f"Loading primary model: {model_name}")
            
            self.corrector_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.corrector_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Grammar correction pipeline
            self.grammar_pipeline = pipeline(
                "text2text-generation", 
                model="vennify/t5-base-grammar-correction",
                device=0 if self.device.type == "cuda" else -1
            )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def detect_errors(self, text: str) -> Dict[str, float]:
        """
        Detect potential errors in ASR output.
        
        Args:
            text: Raw ASR output text
            
        Returns:
            Dictionary of error types and their confidence scores
        """
        errors = {
            "incomplete_words": 0.0,
            "grammar_issues": 0.0,
            "coherence_score": 0.0,
            "overlap_artifacts": 0.0,
            "confidence": 1.0
        }
        
        if not text or len(text.strip()) == 0:
            errors["confidence"] = 0.0
            return errors
        
        # Check for incomplete words (common in overlapping speech)
        incomplete_patterns = [
            r'\b\w*-\s',      # word- 
            r'\s-\w*\b',      # -word
            r'\b\w{1,2}\b',   # very short words
            r'\w+#+'          # word###
        ]
        
        for pattern in incomplete_patterns:
            matches = len(re.findall(pattern, text))
            errors["incomplete_words"] += matches
        
        # Normalize to 0-1 range
        word_count = max(len(text.split()), 1)
        errors["incomplete_words"] = min(errors["incomplete_words"] / word_count, 1.0)
        
        # Check for overlap artifacts
        overlap_patterns = [
            r'(\w+)\s+\1{2,}',  # Repeated words
            r'[^\w\s]{3,}',     # Multiple non-word characters
            r'\b[A-Z]{2,}\b',   # All caps (shouting detection)
        ]
        
        for pattern in overlap_patterns:
            if re.search(pattern, text):
                errors["overlap_artifacts"] += 0.3
        
        errors["overlap_artifacts"] = min(errors["overlap_artifacts"], 1.0)
        
        # Calculate overall confidence
        total_errors = sum([errors[k] for k in errors if k != "confidence"])
        errors["confidence"] = max(0.0, 1.0 - (total_errors / 4.0))
        
        return errors
    
    def correct_segment(self, segment: TranscriptionSegment) -> TranscriptionSegment:
        """
        Main correction method for a transcription segment.
        
        Args:
            segment: TranscriptionSegment to correct
            
        Returns:
            Updated TranscriptionSegment with corrected text
        """
        # Check cache first
        cache_key = self._get_cache_key(segment)
        if cache_key in self.correction_cache:
            segment.corrected_text = self.correction_cache[cache_key]
            return segment
        
        # Detect errors
        error_scores = self.detect_errors(segment.raw_text)
        segment.correction_metadata = error_scores
        
        # Skip correction if confidence is high
        if error_scores["confidence"] > self.config.get('correction_threshold', 0.8):
            segment.corrected_text = segment.raw_text
            return segment
        
        # Apply correction based on error severity
        if error_scores["confidence"] < 0.3:
            # Severe corruption - reconstruct from context
            corrected = self._reconstruct_from_context(segment)
        else:
            # Standard correction
            corrected = self._apply_correction(segment)
        
        # Post-process with grammar correction
        corrected = self._apply_grammar_correction(corrected)
        
        # Update segment and cache
        segment.corrected_text = corrected
        self.correction_cache[cache_key] = corrected
        
        # Update context buffer
        self._update_context_buffer(corrected)
        
        return segment
    
    def _get_cache_key(self, segment: TranscriptionSegment) -> str:
        """Generate cache key for segment"""
        context_str = ":".join(segment.context_window[-2:]) if segment.context_window else ""
        return f"{segment.raw_text}:{context_str}"
    
    def _apply_correction(self, segment: TranscriptionSegment) -> str:
        """
        Apply standard correction using the primary model.
        
        Args:
            segment: TranscriptionSegment to correct
            
        Returns:
            Corrected text
        """
        try:
            # Prepare input with context
            if segment.context_window and len(segment.context_window) > 0:
                context_str = " ".join(segment.context_window[-3:])
                input_text = f"Context: {context_str} Current: {segment.raw_text}"
            else:
                input_text = f"Correct the transcription errors: {segment.raw_text}"
            
            # Tokenize
            inputs = self.corrector_tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            # Generate correction
            with torch.no_grad():
                outputs = self.corrector_model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    early_stopping=True
                )
            
            corrected = self.corrector_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected.strip()
            
        except Exception as e:
            logger.error(f"Correction failed: {e}")
            return segment.raw_text
    
    def _reconstruct_from_context(self, segment: TranscriptionSegment) -> str:
        """
        Reconstruct severely corrupted text using context.
        
        Args:
            segment: TranscriptionSegment with corrupted text
            
        Returns:
            Reconstructed text
        """
        # Extract any salvageable words
        words = segment.raw_text.split()
        valid_words = [w for w in words if len(w) > 2 and re.match(r'^[a-zA-Z]+$', w)]
        
        if not segment.context_window or len(segment.context_window) == 0:
            # No context available - try basic reconstruction
            prompt = f"Complete this partial sentence: {' '.join(valid_words)}"
        else:
            # Use context for guided reconstruction
            context_str = " ".join(segment.context_window[-2:])
            prompt = f"Given the context '{context_str}', complete: {' '.join(valid_words)}"
        
        try:
            inputs = self.corrector_tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.corrector_model.generate(
                    **inputs,
                    max_length=100,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
            
            reconstructed = self.corrector_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return reconstructed.strip()
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            return " ".join(valid_words) if valid_words else segment.raw_text
    
    def _apply_grammar_correction(self, text: str) -> str:
        """Apply grammar correction as post-processing"""
        try:
            if len(text.strip()) == 0:
                return text
                
            result = self.grammar_pipeline(text, max_length=512)
            return result[0]['generated_text']
        except Exception as e:
            logger.warning(f"Grammar correction failed: {e}")
            return text
    
    def _update_context_buffer(self, corrected_text: str):
        """Update context buffer with corrected text"""
        self.context_buffer.append(corrected_text)
        
        # Keep only recent context
        max_size = self.config.get('context_window_size', 5)
        if len(self.context_buffer) > max_size:
            self.context_buffer.pop(0)
    
    def get_context_window(self) -> List[str]:
        """Get current context window"""
        return self.context_buffer.copy()
    
    def clear_cache(self):
        """Clear correction cache"""
        self.correction_cache.clear()
        logger.info("Correction cache cleared")