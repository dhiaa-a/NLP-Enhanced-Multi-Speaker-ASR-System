#!/usr/bin/env python3
"""
Demo script for the Multi-Speaker ASR with NLP Enhancement system.
Shows before/after comparisons and real-time processing capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
from typing import List
import time
from colorama import init, Fore, Back, Style
from src.pipeline.orchestrator import MultiSpeakerASRPipeline, AudioSegment
from src.nlp.error_corrector import TranscriptionSegment
from src.nlp.correction_strategies import CorrectionStrategies


# Initialize colorama for colored output
init()

class ASRDemo:
    """Demo class for showcasing the system capabilities"""
    
    def __init__(self):
        print(f"{Fore.CYAN}Initializing Multi-Speaker ASR System...{Style.RESET_ALL}")
        self.pipeline = MultiSpeakerASRPipeline()
        print(f"{Fore.GREEN}✓ System initialized successfully{Style.RESET_ALL}\n")
    
    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{Back.BLUE}{Fore.WHITE} {text} {Style.RESET_ALL}")
        print("=" * 60)
    
    def print_comparison(self, segment: TranscriptionSegment):
        """Print before/after comparison"""
        print(f"\n{Fore.YELLOW}Speaker {segment.speaker_id}{Style.RESET_ALL} "
              f"({segment.start_time:.1f}s - {segment.end_time:.1f}s)")
        
        print(f"{Fore.RED}Raw ASR:{Style.RESET_ALL} {segment.raw_text}")
        print(f"{Fore.GREEN}Corrected:{Style.RESET_ALL} {segment.corrected_text}")
        
        if segment.correction_metadata:
            confidence = segment.correction_metadata.get('confidence', 0)
            print(f"{Fore.CYAN}Confidence:{Style.RESET_ALL} {confidence:.2f}")
    
    async def demo_simulated_audio(self):
        """Demo with simulated garbled ASR output"""
        self.print_header("DEMO 1: Simulated Overlapping Speech Correction")
        
        # Simulate garbled ASR outputs (typical of overlapping speech)
        test_cases = [
            {
                "speaker_id": 1,
                "raw": "So I was think- the meet- tomorrow at tw- no actually thre-",
                "timestamp": 0.0,
                "duration": 3.0
            },
            {
                "speaker_id": 2,
                "raw": "Yeah I agr## with tha+ but we should also consid## the budg##",
                "timestamp": 3.0,
                "duration": 3.5
            },
            {
                "speaker_id": 1,
                "raw": "Exact### and the the the deadl### is next fri###",
                "timestamp": 6.5,
                "duration": 2.5
            },
            {
                "speaker_id": 3,
                "raw": "Can you he## me? The connec### is really b##",
                "timestamp": 9.0,
                "duration": 2.0
            }
        ]
        
        print(f"{Fore.CYAN}Processing {len(test_cases)} segments with simulated ASR errors...{Style.RESET_ALL}")
        
        for test in test_cases:
            # Create a mock audio segment
            audio_data = np.random.randn(int(test["duration"] * 16000)).astype(np.float32)
            
            # Directly test NLP correction without full pipeline
            segment = TranscriptionSegment(
                speaker_id=test["speaker_id"],
                start_time=test["timestamp"],
                end_time=test["timestamp"] + test["duration"],
                raw_text=test["raw"]
            )
            
            # Apply correction
            start_time = time.time()
            strategies = CorrectionStrategies()
            segment.corrected_text = strategies.pattern_based_correction(segment.raw_text)
            latency_ms = (time.time() - start_time) * 1000
            
            # Display results
            self.print_comparison(segment)
            print(f"{Fore.MAGENTA}Processing time:{Style.RESET_ALL} {latency_ms:.1f}ms")
            print("-" * 60)
    
    async def demo_audio_file(self, audio_file: str = None):
        """Demo with actual audio file"""
        self.print_header("DEMO 2: Audio File Processing")
        
        if not audio_file or not os.path.exists(audio_file):
            print(f"{Fore.YELLOW}No audio file provided. Using simulated audio.{Style.RESET_ALL}")
            # Create simulated audio
            audio_file = self._create_simulated_audio()
        
        print(f"Processing audio file: {audio_file}")
        
        try:
            # Process the file
            start_time = time.time()
            segments = self.pipeline.process_file(audio_file)
            total_time = (time.time() - start_time) * 1000
            
            print(f"\n{Fore.GREEN}✓ Processed {len(segments)} speaker segments{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Total processing time:{Style.RESET_ALL} {total_time:.1f}ms")
            
            # Display results
            for segment in segments:
                self.print_comparison(segment)
                
        except Exception as e:
            print(f"{Fore.RED}Error processing audio file: {e}{Style.RESET_ALL}")
    
    def _create_simulated_audio(self) -> str:
        """Create a simulated audio file for testing"""
        import soundfile as sf
        
        # Generate 10 seconds of white noise
        duration = 10  # seconds
        sample_rate = 16000
        audio = np.random.randn(duration * sample_rate) * 0.1
        
        # Add some structure (sine waves at different frequencies)
        t = np.linspace(0, duration, duration * sample_rate)
        audio += 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        audio += 0.2 * np.sin(2 * np.pi * 554 * t)  # C#5 note
        
        # Save to temporary file
        temp_file = "temp_demo_audio.wav"
        sf.write(temp_file, audio, sample_rate)
        
        return temp_file
    
    async def demo_real_time_stream(self):
        """Demo real-time streaming capability"""
        self.print_header("DEMO 3: Real-Time Streaming Simulation")
        
        print(f"{Fore.CYAN}Simulating real-time audio stream...{Style.RESET_ALL}")
        print("(Press Ctrl+C to stop)\n")
        
        async def audio_generator():
            """Generate audio segments simulating real-time stream"""
            chunk_duration = 0.5  # 500ms chunks
            chunk_samples = int(chunk_duration * 16000)
            timestamp = 0.0
            
            # Simulated ASR outputs
            simulated_texts = [
                "Hello every### welcome to the meet-",
                "Today we'll disc### the new proj###",
                "First let me shar- my scre##",
                "Can everyone see### my present###?",
                "Great so the the main object###",
                "is to improv### our custom## satisf###"
            ]
            
            for i in range(len(simulated_texts)):
                # Generate audio chunk
                audio_data = np.random.randn(chunk_samples).astype(np.float32)
                
                segment = AudioSegment(
                    data=audio_data,
                    sample_rate=16000,
                    timestamp=timestamp,
                    duration=chunk_duration
                )
                
                timestamp += chunk_duration
                yield segment
                
                # Simulate real-time delay
                await asyncio.sleep(chunk_duration)
        
        try:
            segment_count = 0
            async for result in self.pipeline.process_audio_stream(audio_generator()):
                segment_count += 1
                print(f"\n{Fore.YELLOW}[Segment {segment_count}]{Style.RESET_ALL}")
                
                for segment in result.segments:
                    self.print_comparison(segment)
                
                print(f"{Fore.MAGENTA}Latency:{Style.RESET_ALL} {result.total_latency_ms:.1f}ms")
                
                # Show stage latencies
                print(f"{Fore.CYAN}Stage latencies:{Style.RESET_ALL}")
                for stage, latency in result.stage_latencies.items():
                    print(f"  - {stage}: {latency:.1f}ms")
                    
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stream stopped by user{Style.RESET_ALL}")
    
    def show_performance_metrics(self):
        """Display performance metrics"""
        self.print_header("Performance Metrics")
        
        metrics = self.pipeline.get_performance_metrics()
        
        print(f"{Fore.CYAN}Pipeline Metrics:{Style.RESET_ALL}")
        print(f"  - Processed segments: {metrics['processed_segments']}")
        print(f"  - Average latency: {metrics['average_latency_ms']:.1f}ms")
        print(f"  - Errors: {metrics['error_count']}")
        
        if metrics['stage_latencies']:
            print(f"\n{Fore.CYAN}Average Stage Latencies:{Style.RESET_ALL}")
            for stage, latency in metrics['stage_latencies'].items():
                print(f"  - {stage}: {latency:.1f}ms")
        
        nlp_stats = metrics.get('nlp_correction_stats', {})
        if nlp_stats:
            print(f"\n{Fore.CYAN}NLP Correction Stats:{Style.RESET_ALL}")
            print(f"  - Total corrections: {nlp_stats.get('total_corrections', 0)}")
            print(f"  - Cache hit rate: {nlp_stats.get('cache_hit_rate', 0):.1%}")
            print(f"  - Deadline miss rate: {nlp_stats.get('deadline_miss_rate', 0):.1%}")
            
            if 'strategy_usage' in nlp_stats:
                print(f"\n{Fore.CYAN}Strategy Usage:{Style.RESET_ALL}")
                for strategy, count in nlp_stats['strategy_usage'].items():
                    print(f"  - {strategy}: {count}")

async def main():
    """Main demo function"""
    print(f"\n{Back.GREEN}{Fore.BLACK} Multi-Speaker ASR with NLP Enhancement Demo {Style.RESET_ALL}\n")
    
    demo = ASRDemo()
    
    try:
        # Run demos
        await demo.demo_simulated_audio()
        
        # Uncomment to test with audio file
        # await demo.demo_audio_file("path/to/your/audio.wav")
        
        # Real-time streaming demo
        await demo.demo_real_time_stream()
        
    except Exception as e:
        print(f"\n{Fore.RED}Demo error: {e}{Style.RESET_ALL}")
    
    finally:
        # Show performance metrics
        demo.show_performance_metrics()
        
        # Cleanup
        demo.pipeline.stop()
        print(f"\n{Fore.GREEN}Demo completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())