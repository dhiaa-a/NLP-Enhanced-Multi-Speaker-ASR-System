#!/usr/bin/env python3
"""
Basic usage examples for the Multi-Speaker ASR with NLP Enhancement system.
Shows common use cases and code snippets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import asyncio
from pathlib import Path

# Import the main components
from src.pipeline.orchestrator import MultiSpeakerASRPipeline, AudioSegment
from src.nlp.error_corrector import NLPErrorCorrector, TranscriptionSegment
from src.nlp.realtime_corrector import RealTimeErrorCorrector
from src.nlp.speaker_models import SpeakerAwareCorrector

def example_1_basic_file_processing():
    """Example 1: Process a single audio file"""
    print("\n=== Example 1: Basic File Processing ===\n")
    
    # Initialize the pipeline
    pipeline = MultiSpeakerASRPipeline()
    
    # Process an audio file
    audio_file = "examples/sample_audio/meeting.wav"  # Replace with your file
    
    if os.path.exists(audio_file):
        # Process the file
        segments = pipeline.process_file(audio_file)
        
        # Print results
        print(f"Processed {len(segments)} speaker segments:\n")
        for segment in segments:
            print(f"Speaker {segment.speaker_id} ({segment.start_time:.1f}s - {segment.end_time:.1f}s):")
            print(f"  Raw:       {segment.raw_text}")
            print(f"  Corrected: {segment.corrected_text}")
            print(f"  Confidence: {segment.confidence:.2f}\n")
    else:
        print(f"Audio file not found: {audio_file}")
        print("Creating a simulated example...")
        
        # Simulate processing
        segment = TranscriptionSegment(
            speaker_id=1,
            start_time=0.0,
            end_time=3.0,
            raw_text="I think we sh## meet tomor- at thre-"
        )
        corrected = pipeline.nlp_corrector.correct_segment(segment)
        print(f"Raw:       {segment.raw_text}")
        print(f"Corrected: {corrected.corrected_text}")
    
    # Cleanup
    pipeline.stop()


def example_2_nlp_correction_only():
    """Example 2: Use NLP correction standalone"""
    print("\n=== Example 2: NLP Correction Only ===\n")
    
    # Initialize just the NLP corrector
    corrector = NLPErrorCorrector()
    
    # Example garbled ASR outputs
    test_cases = [
        "The meet- is sched### for tomor-",
        "Can you he## me? The connec### is b##",
        "Let's disc### the budg## at the next meet###",
        "I agr## with your propos### completely"
    ]
    
    print("Correcting garbled ASR output:\n")
    for raw_text in test_cases:
        # Create a segment
        segment = TranscriptionSegment(
            speaker_id=1,
            start_time=0.0,
            end_time=3.0,
            raw_text=raw_text
        )
        
        # Correct it
        corrected_segment = corrector.correct_segment(segment)
        
        print(f"Raw:       {raw_text}")
        print(f"Corrected: {corrected_segment.corrected_text}")
        print(f"Confidence: {corrected_segment.confidence:.2f}\n")


def example_3_real_time_correction():
    """Example 3: Real-time correction with latency constraints"""
    print("\n=== Example 3: Real-Time Correction ===\n")
    
    # Initialize real-time corrector with 50ms deadline
    realtime_corrector = RealTimeErrorCorrector(max_latency_ms=50)
    
    # Test texts
    test_texts = [
        "Quick fix for meet- tomorrow",
        "The budg## is appro### five mill###",
        "Severely corrupted ### ### ### text ###"
    ]
    
    print("Real-time correction with 50ms deadline:\n")
    for text in test_texts:
        result = realtime_corrector.correct_with_deadline(text)
        
        print(f"Raw:       {text}")
        print(f"Corrected: {result.text}")
        print(f"Strategy:  {result.strategy}")
        print(f"Latency:   {result.latency_ms:.1f}ms")
        print(f"Confidence: {result.confidence:.2f}\n")
    
    # Show performance stats
    stats = realtime_corrector.get_performance_stats()
    print("Performance Statistics:")
    print(f"  Total corrections: {stats['total_corrections']}")
    print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
    print(f"  Deadline miss rate: {stats.get('deadline_miss_rate', 0):.1%}")


def example_4_speaker_aware_correction():
    """Example 4: Speaker-specific correction"""
    print("\n=== Example 4: Speaker-Aware Correction ===\n")
    
    # Initialize speaker-aware corrector
    speaker_corrector = SpeakerAwareCorrector()
    
    # Simulate learning from Speaker 1 (technical person)
    print("Training Speaker 1 (technical vocabulary):")
    technical_utterances = [
        "We need to optimize the algorithm performance",
        "The database query is too slow",
        "Let's implement a caching layer",
        "The API response time is critical"
    ]
    
    for utterance in technical_utterances:
        speaker_corrector.update_speaker_model(1, utterance)
    
    # Simulate learning from Speaker 2 (business person)
    print("Training Speaker 2 (business vocabulary):")
    business_utterances = [
        "We need to increase our market share",
        "The quarterly revenue looks promising",
        "Let's discuss the budget allocation",
        "Customer satisfaction is our priority"
    ]
    
    for utterance in business_utterances:
        speaker_corrector.update_speaker_model(2, utterance)
    
    # Test corrections
    print("\nTesting speaker-specific corrections:\n")
    
    # Technical term corruption - should work better for Speaker 1
    tech_corrupted = "We need to optim### the algor###"
    print(f"Technical text: {tech_corrupted}")
    print(f"Speaker 1 correction: {speaker_corrector.speaker_specific_correction(tech_corrupted, 1)}")
    print(f"Speaker 2 correction: {speaker_corrector.speaker_specific_correction(tech_corrupted, 2)}")
    
    print()
    
    # Business term corruption - should work better for Speaker 2
    business_corrupted = "The quart### reven### looks good"
    print(f"Business text: {business_corrupted}")
    print(f"Speaker 1 correction: {speaker_corrector.speaker_specific_correction(business_corrupted, 1)}")
    print(f"Speaker 2 correction: {speaker_corrector.speaker_specific_correction(business_corrupted, 2)}")
    
    # Show speaker statistics
    print("\nSpeaker Statistics:")
    for speaker_id in [1, 2]:
        stats = speaker_corrector.get_speaker_statistics(speaker_id)
        print(f"\nSpeaker {speaker_id}:")
        print(f"  Vocabulary size: {stats['vocabulary_size']}")
        print(f"  Average utterance length: {stats['average_utterance_length']:.1f} words")
        print(f"  Most frequent words: {stats['most_frequent_words'][:5]}")


async def example_5_streaming_processing():
    """Example 5: Process streaming audio"""
    print("\n=== Example 5: Streaming Audio Processing ===\n")
    
    # Initialize pipeline
    pipeline = MultiSpeakerASRPipeline()
    
    async def simulate_audio_stream():
        """Simulate an audio stream"""
        chunk_duration = 0.5  # 500ms chunks
        sample_rate = 16000
        
        # Simulate 5 chunks
        for i in range(5):
            # Generate random audio (in practice, this would be real audio)
            audio_data = np.random.randn(int(chunk_duration * sample_rate)).astype(np.float32)
            
            yield AudioSegment(
                data=audio_data,
                sample_rate=sample_rate,
                timestamp=i * chunk_duration,
                duration=chunk_duration
            )
            
            # Simulate real-time delay
            await asyncio.sleep(chunk_duration)
    
    print("Processing streaming audio (5 chunks)...\n")
    
    chunk_count = 0
    async for result in pipeline.process_audio_stream(simulate_audio_stream()):
        chunk_count += 1
        print(f"Chunk {chunk_count}:")
        print(f"  Latency: {result.total_latency_ms:.1f}ms")
        print(f"  Segments: {len(result.segments)}")
        
        for segment in result.segments:
            print(f"  Speaker {segment.speaker_id}: {segment.corrected_text}")
        print()
    
    # Cleanup
    pipeline.stop()


def example_6_batch_processing():
    """Example 6: Batch process multiple texts"""
    print("\n=== Example 6: Batch Processing ===\n")
    
    # Initialize real-time corrector
    corrector = RealTimeErrorCorrector()
    
    # Batch of corrupted texts
    texts = [
        "Meeting at thre### tomorrow",
        "Budget approv### for Q4",
        "New proj### deadline Friday",
        "Team perform### review needed",
        "Client satisf### survey results"
    ]
    
    print("Batch processing 5 texts:\n")
    
    # Process batch
    results = corrector.batch_correct(texts)
    
    # Display results
    for text, result in zip(texts, results):
        print(f"Raw:       {text}")
        print(f"Corrected: {result.text}")
        print(f"Latency:   {result.latency_ms:.1f}ms\n")
    
    # Calculate average performance
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    print(f"Average latency: {avg_latency:.1f}ms")


def example_7_save_and_load_profiles():
    """Example 7: Save and load speaker profiles"""
    print("\n=== Example 7: Save/Load Speaker Profiles ===\n")
    
    # Create pipeline
    pipeline = MultiSpeakerASRPipeline()
    
    # Train some speaker profiles
    print("Training speaker profiles...")
    
    # Speaker 1
    for text in ["Hello, this is speaker one", "I like to discuss technical topics"]:
        pipeline.speaker_corrector.update_speaker_model(1, text)
    
    # Speaker 2
    for text in ["Hi, I'm speaker two", "Let's talk about business strategy"]:
        pipeline.speaker_corrector.update_speaker_model(2, text)
    
    # Save profiles
    profile_path = "speaker_profiles.pkl"
    pipeline.save_speaker_profiles(profile_path)
    print(f"Saved profiles to {profile_path}")
    
    # Create new pipeline and load profiles
    new_pipeline = MultiSpeakerASRPipeline()
    new_pipeline.load_speaker_profiles(profile_path)
    print(f"Loaded profiles from {profile_path}")
    
    # Verify profiles loaded correctly
    for speaker_id in [1, 2]:
        stats = new_pipeline.speaker_corrector.get_speaker_statistics(speaker_id)
        print(f"\nSpeaker {speaker_id} vocabulary size: {stats['vocabulary_size']}")
    
    # Cleanup
    pipeline.stop()
    new_pipeline.stop()
    
    # Remove test file
    if os.path.exists(profile_path):
        os.remove(profile_path)


def main():
    """Run all examples"""
    examples = [
        example_1_basic_file_processing,
        example_2_nlp_correction_only,
        example_3_real_time_correction,
        example_4_speaker_aware_correction,
        # example_5_streaming_processing,  # Async example
        example_6_batch_processing,
        example_7_save_and_load_profiles
    ]
    
    print("\n" + "="*60)
    print("Multi-Speaker ASR with NLP Enhancement - Usage Examples")
    print("="*60)
    
    for example in examples:
        try:
            if asyncio.iscoroutinefunction(example):
                asyncio.run(example())
            else:
                example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
        
        print("\n" + "-"*60)
    
    # Run async example separately
    print("\nRunning async streaming example...")
    try:
        asyncio.run(example_5_streaming_processing())
    except Exception as e:
        print(f"Error in streaming example: {e}")
    
    print("\n✅ All examples completed!")
    print("\nFor more examples, see the documentation and test files.")


if __name__ == "__main__":
    main()