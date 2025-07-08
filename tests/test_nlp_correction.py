"""
Tests for NLP Error Correction Module
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp.error_corrector import NLPErrorCorrector, TranscriptionSegment
from src.nlp.correction_strategies import CorrectionStrategies
from src.nlp.realtime_corrector import RealTimeErrorCorrector

class TestNLPErrorCorrector:
    """Test the main NLP error corrector"""
    
    @pytest.fixture
    def corrector(self):
        """Create a corrector instance for testing"""
        # Use test config
        return NLPErrorCorrector()
    
    def test_detect_errors(self, corrector):
        """Test error detection in ASR output"""
        # Test incomplete words
        text = "I think we sh## meet tomor- at thre-"
        errors = corrector.detect_errors(text)
        
        assert errors['incomplete_words'] > 0
        assert errors['confidence'] < 1.0
    
    def test_correct_segment(self, corrector):
        """Test segment correction"""
        segment = TranscriptionSegment(
            speaker_id=1,
            start_time=0.0,
            end_time=3.0,
            raw_text="The meet- tomorrow at tw-"
        )
        
        corrected_segment = corrector.correct_segment(segment)
        
        assert corrected_segment.corrected_text is not None
        assert "-" not in corrected_segment.corrected_text
        assert "meet" in corrected_segment.corrected_text.lower()
    
    def test_context_aware_correction(self, corrector):
        """Test correction with context"""
        # Set up context
        corrector.context_buffer = [
            "We need to schedule our next team meeting",
            "When works best for everyone?"
        ]
        
        segment = TranscriptionSegment(
            speaker_id=1,
            start_time=5.0,
            end_time=8.0,
            raw_text="I think tomor### at thre###",
            context_window=corrector.get_context_window()
        )
        
        corrected_segment = corrector.correct_segment(segment)
        
        # Should correctly infer "tomorrow" and "three" from context
        assert "tomorrow" in corrected_segment.corrected_text.lower()

class TestCorrectionStrategies:
    """Test individual correction strategies"""
    
    @pytest.fixture
    def strategies(self):
        """Create strategies instance"""
        return CorrectionStrategies()
    
    def test_pattern_based_correction(self, strategies):
        """Test pattern-based corrections"""
        # Test incomplete words
        text = "meet- tomorrow"
        corrected = strategies.pattern_based_correction(text)
        assert corrected == "meet tomorrow"
        
        # Test corrupted words
        text = "budg### is ready"
        corrected = strategies.pattern_based_correction(text)
        assert "###" not in corrected
        
        # Test repetitions
        text = "the the the meeting"
        corrected = strategies.pattern_based_correction(text)
        assert corrected == "the meeting"
    
    def test_phonetic_correction(self, strategies):
        """Test phonetic similarity correction"""
        text = "thre people"
        corrected = strategies.phonetic_correction(text)
        assert "three" in corrected
        
        text = "tw meetings"
        corrected = strategies.phonetic_correction(text)
        assert "two" in corrected
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_masked_lm_correction(self, strategies):
        """Test BERT-based masked language model correction"""
        text = "The ### is tomorrow"
        corrected = strategies.masked_language_model_correction(text)
        assert "###" not in corrected

class TestRealTimeCorrector:
    """Test real-time error corrector"""
    
    @pytest.fixture
    def realtime_corrector(self):
        """Create real-time corrector with tight deadline"""
        return RealTimeErrorCorrector(max_latency_ms=50)
    
    def test_deadline_compliance(self, realtime_corrector):
        """Test that correction stays within deadline"""
        text = "I think we sh## meet tomor- at thre-"
        
        result = realtime_corrector.correct_with_deadline(text)
        
        assert result.latency_ms <= 50
        assert result.text is not None
        assert result.strategy in ['ultra_fast', 'fast_pattern', 'neural_small', 'cache']
    
    def test_cache_functionality(self, realtime_corrector):
        """Test caching for repeated corrections"""
        text = "meet- tomorrow"
        
        # First call
        result1 = realtime_corrector.correct_with_deadline(text)
        assert result1.strategy != 'cache'
        
        # Second call should hit cache
        result2 = realtime_corrector.correct_with_deadline(text)
        assert result2.strategy == 'cache'
        assert result2.latency_ms < result1.latency_ms
    
    def test_progressive_strategies(self, realtime_corrector):
        """Test that strategies are tried progressively"""
        # Very corrupted text should trigger multiple strategies
        text = "I ### we ### ### tomor### ### ###"
        
        result = realtime_corrector.correct_with_deadline(text)
        
        # Should have tried to improve the text
        assert result.text != text
        assert result.confidence < 0.5  # Low confidence expected
    
    @pytest.mark.asyncio
    async def test_async_correction(self, realtime_corrector):
        """Test async correction interface"""
        text = "meet- tomorrow"
        
        result = await realtime_corrector.correct_async(text)
        
        assert result.text is not None
        assert "-" not in result.text

class TestIntegration:
    """Integration tests for the complete correction pipeline"""
    
    def test_severe_corruption_handling(self):
        """Test handling of severely corrupted text"""
        corrector = NLPErrorCorrector()
        
        segment = TranscriptionSegment(
            speaker_id=1,
            start_time=0.0,
            end_time=3.0,
            raw_text="### ### ### ### ###"
        )
        
        corrected_segment = corrector.correct_segment(segment)
        
        # Should provide some output even for severe corruption
        assert corrected_segment.corrected_text is not None
        assert len(corrected_segment.corrected_text) > 0
    
    def test_multiple_speakers(self):
        """Test correction with multiple speakers"""
        from src.nlp.speaker_models import SpeakerAwareCorrector
        
        speaker_corrector = SpeakerAwareCorrector()
        
        # Train on speaker 1
        for i in range(10):
            speaker_corrector.update_speaker_model(
                speaker_id=1,
                corrected_text="I think we should meet tomorrow at three"
            )
        
        # Test correction for speaker 1
        corrected = speaker_corrector.speaker_specific_correction(
            "I think we sh### meet tomor###",
            speaker_id=1
        )
        
        assert "should" in corrected
        assert "tomorrow" in corrected

# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for correction strategies"""
    
    def test_correction_speed(self, benchmark):
        """Benchmark correction speed"""
        corrector = RealTimeErrorCorrector(max_latency_ms=100)
        text = "I think we sh## meet tomor- at thre-"
        
        result = benchmark(corrector.correct_with_deadline, text)
        assert result.latency_ms < 100
    
    def test_batch_processing(self, benchmark):
        """Benchmark batch processing"""
        corrector = RealTimeErrorCorrector()
        texts = [
            "meet- tomorrow",
            "budg### ready", 
            "the the meeting",
            "connec### bad"
        ] * 10
        
        results = benchmark(corrector.batch_correct, texts)
        assert len(results) == len(texts)

import torch