"""
Real-time Error Corrector Module
Optimized for low-latency processing with fallback strategies.
"""

import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional, Tuple
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import asyncio
from loguru import logger
from dataclasses import dataclass
import heapq

@dataclass
class CorrectionResult:
    """Result from a correction attempt"""
    text: str
    latency_ms: float
    strategy: str
    confidence: float

class RealTimeErrorCorrector:
    """
    Optimized error corrector for real-time processing.
    Uses multiple strategies with deadline-aware execution.
    """
    
    def __init__(self, max_latency_ms: int = 100, config: dict = None):
        self.max_latency_ms = max_latency_ms
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._init_fast_models()
        
        # Cache for common corrections
        self.cache = {}
        self.cache_size = self.config.get('cache_size', 1000)
        
        # Pre-compiled patterns for speed
        self.compiled_patterns = self._compile_patterns()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.performance_stats = {
            'total_corrections': 0,
            'cache_hits': 0,
            'deadline_misses': 0,
            'strategy_usage': {}
        }
        
        logger.info(f"Real-time corrector initialized (max latency: {max_latency_ms}ms)")
    
    def _init_fast_models(self):
        """Initialize lightweight models for speed"""
        try:
            # Use T5-small for speed
            model_name = self.config.get('fast_model', 't5-small')
            self.fast_model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.fast_tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            # Move to GPU if available
            self.fast_model = self.fast_model.to(self.device)
            self.fast_model.eval()
            
            # Optimize for inference
            if self.device.type == "cuda":
                self.fast_model = self.fast_model.half()  # FP16 for speed
            
        except Exception as e:
            logger.error(f"Failed to load fast model: {e}")
            self.fast_model = None
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile regex patterns for speed"""
        return {
            'incomplete_word': re.compile(r'(\w{3,})-\s*|\s*-(\w{3,})'),
            'corrupted': re.compile(r'(\w+)##+'),
            'repetition': re.compile(r'\b(\w+)(\s+\1){2,}\b'),
            'fragments': re.compile(r'\b\w{1,2}\b'),
            'noise': re.compile(r'[^\w\s]{3,}'),
            'multi_space': re.compile(r'\s+'),
        }
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, text: str, context_hash: str) -> str:
        """Generate cache key"""
        return f"{hash(text)}:{context_hash}"
    
    def correct_with_deadline(self, text: str, context: Optional[List[str]] = None) -> CorrectionResult:
        """
        Correct text within deadline, using progressively faster strategies.
        
        Args:
            text: Raw text to correct
            context: Optional context
            
        Returns:
            CorrectionResult with corrected text and metadata
        """
        start_time = time.time()
        self.performance_stats['total_corrections'] += 1
        
        # Check cache first
        context_hash = hash(tuple(context[-2:])) if context else ""
        cache_key = self._get_cache_key(text, str(context_hash))
        
        if cache_key in self.cache:
            self.performance_stats['cache_hits'] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            return CorrectionResult(
                text=self.cache[cache_key],
                latency_ms=elapsed_ms,
                strategy="cache",
                confidence=1.0
            )
        
        # Define correction strategies in order of speed
        strategies = [
            ("ultra_fast", self.ultra_fast_correction, 10),   # 10ms budget
            ("fast_pattern", self.fast_pattern_correction, 30), # 30ms budget
            ("neural_small", self.fast_neural_correction, 70),  # 70ms budget
        ]
        
        best_result = None
        remaining_budget = self.max_latency_ms
        
        for strategy_name, strategy_func, strategy_budget in strategies:
            # Check if we have time for this strategy
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= self.max_latency_ms * 0.9:  # 90% of budget used
                break
            
            try:
                # Run strategy with timeout
                strategy_start = time.time()
                
                if strategy_name == "neural_small" and context:
                    corrected = strategy_func(text, context)
                else:
                    corrected = strategy_func(text)
                
                strategy_time = (time.time() - strategy_start) * 1000
                
                # Create result
                result = CorrectionResult(
                    text=corrected,
                    latency_ms=strategy_time,
                    strategy=strategy_name,
                    confidence=self._calculate_confidence(text, corrected)
                )
                
                # Update best result if this is better
                if not best_result or result.confidence > best_result.confidence:
                    best_result = result
                
                # Update stats
                self.performance_stats['strategy_usage'][strategy_name] = \
                    self.performance_stats['strategy_usage'].get(strategy_name, 0) + 1
                
                # If confidence is high enough, stop trying other strategies
                if result.confidence > 0.8:
                    break
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        # Use best result or fallback
        if not best_result:
            best_result = CorrectionResult(
                text=self.ultra_fast_correction(text),
                latency_ms=(time.time() - start_time) * 1000,
                strategy="fallback",
                confidence=0.3
            )
        
        # Update cache if we're under budget
        total_time_ms = (time.time() - start_time) * 1000
        if total_time_ms < self.max_latency_ms and len(self.cache) < self.cache_size:
            self.cache[cache_key] = best_result.text
        elif total_time_ms >= self.max_latency_ms:
            self.performance_stats['deadline_misses'] += 1
        
        best_result.latency_ms = total_time_ms
        return best_result
    
    def ultra_fast_correction(self, text: str) -> str:
        """
        Ultra-fast pattern-based correction (< 10ms).
        Only fixes most common and obvious errors.
        """
        if not text:
            return text
        
        # Apply only the most critical corrections
        corrected = text
        
        # Remove word fragments at end
        corrected = self.compiled_patterns['incomplete_word'].sub(r'\1\2', corrected)
        
        # Remove simple repetitions
        corrected = self.compiled_patterns['repetition'].sub(r'\1', corrected)
        
        # Clean excessive spaces
        corrected = self.compiled_patterns['multi_space'].sub(' ', corrected)
        
        return corrected.strip()
    
    def fast_pattern_correction(self, text: str) -> str:
        """
        Fast pattern-based correction (< 30ms).
        More comprehensive than ultra-fast.
        """
        if not text:
            return text
        
        corrected = text
        
        # All ultra-fast corrections
        corrected = self.ultra_fast_correction(corrected)
        
        # Additional corrections
        
        # Fix corrupted words
        def quick_fix_corrupted(match):
            partial = match.group(1)
            if len(partial) >= 4:
                return partial
            return ''
        
        corrected = self.compiled_patterns['corrupted'].sub(quick_fix_corrupted, corrected)
        
        # Remove noise artifacts
        corrected = self.compiled_patterns['noise'].sub(' ', corrected)
        
        # Quick word replacements
        quick_fixes = {
            'thre': 'three',
            'tw': 'two',
            'mee': 'meet',
            'tomor': 'tomorrow',
            'yest': 'yesterday'
        }
        
        words = corrected.split()
        fixed_words = []
        for word in words:
            cleaned = word.lower().strip('.,!?')
            if cleaned in quick_fixes:
                fixed_words.append(quick_fixes[cleaned])
            elif len(cleaned) > 2:  # Skip very short fragments
                fixed_words.append(word)
        
        return ' '.join(fixed_words).strip()
    
    def fast_neural_correction(self, text: str, context: Optional[List[str]] = None) -> str:
        """
        Fast neural correction using small model (< 70ms).
        """
        if not self.fast_model or not text:
            return self.fast_pattern_correction(text)
        
        try:
            # Prepare input
            if context and len(context) > 0:
                context_str = " ".join(context[-2:])  # Only last 2 for speed
                input_text = f"fix: {context_str} | {text}"
            else:
                input_text = f"fix: {text}"
            
            # Tokenize with strict length limit
            inputs = self.fast_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=64,  # Short for speed
                truncation=True,
                padding=False
            ).to(self.device)
            
            # Generate with minimal beam search
            with torch.no_grad():
                outputs = self.fast_model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=2,  # Minimal beams
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=False,  # Deterministic for speed
                    use_cache=True
                )
            
            corrected = self.fast_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected.strip()
            
        except Exception as e:
            logger.warning(f"Fast neural correction failed: {e}")
            return self.fast_pattern_correction(text)
    
    def _calculate_confidence(self, original: str, corrected: str) -> float:
        """Calculate confidence score for correction"""
        if not original or not corrected:
            return 0.0
        
        # Basic confidence based on changes made
        original_words = original.lower().split()
        corrected_words = corrected.lower().split()
        
        if not original_words:
            return 0.0
        
        # Calculate similarity
        common_words = set(original_words).intersection(set(corrected_words))
        similarity = len(common_words) / len(original_words)
        
        # Check for artifacts in corrected text
        artifact_penalty = 0.0
        if '#' in corrected or '-' in corrected:
            artifact_penalty = 0.3
        
        # Length ratio check
        len_ratio = len(corrected_words) / max(len(original_words), 1)
        if len_ratio < 0.5 or len_ratio > 2.0:
            artifact_penalty += 0.2
        
        confidence = similarity - artifact_penalty
        return max(0.0, min(1.0, confidence))
    
    async def correct_async(self, text: str, context: Optional[List[str]] = None) -> CorrectionResult:
        """Async version of correct_with_deadline"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.correct_with_deadline,
            text,
            context
        )
    
    def batch_correct(self, texts: List[str], context: Optional[List[str]] = None) -> List[CorrectionResult]:
        """
        Correct multiple texts in batch for efficiency.
        """
        results = []
        
        # Process in parallel with thread pool
        futures = []
        for text in texts:
            future = self.executor.submit(self.correct_with_deadline, text, context)
            futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                result = future.result(timeout=self.max_latency_ms / 1000)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch correction failed: {e}")
                # Fallback result
                results.append(CorrectionResult(
                    text=self.ultra_fast_correction(text),
                    latency_ms=self.max_latency_ms,
                    strategy="error_fallback",
                    confidence=0.1
                ))
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate rates
        if stats['total_corrections'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_corrections']
            stats['deadline_miss_rate'] = stats['deadline_misses'] / stats['total_corrections']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['deadline_miss_rate'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """Clear the correction cache"""
        self.cache.clear()
        logger.info("Correction cache cleared")
    
    def warmup(self):
        """Warmup the models for better latency"""
        logger.info("Warming up models...")
        
        # Run a few dummy corrections
        test_texts = [
            "This is a test###",
            "Hello wor- how are y##",
            "The meet- tomorrow at thre-"
        ]
        
        for text in test_texts:
            self.correct_with_deadline(text)
        
        logger.info("Warmup complete")