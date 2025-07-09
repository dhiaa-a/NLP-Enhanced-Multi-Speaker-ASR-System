#!/usr/bin/env python3
"""
Performance benchmarking script for Multi-Speaker ASR with NLP Enhancement.
Tests system performance under various conditions and generates reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import asyncio
from loguru import logger
import argparse

from src.pipeline.orchestrator import MultiSpeakerASRPipeline, AudioSegment
from src.nlp.error_corrector import TranscriptionSegment
from src.nlp.correction_strategies import CorrectionStrategies


class ASRBenchmark:
    """Comprehensive benchmarking for the ASR system"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline
        logger.info("Initializing ASR pipeline for benchmarking...")
        self.pipeline = MultiSpeakerASRPipeline()
        
        # Benchmark results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        import platform
        import torch
        import psutil
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
        }
    
    def generate_test_audio(self, duration: float, num_speakers: int = 2) -> np.ndarray:
        """Generate test audio with simulated speakers"""
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Generate base audio
        audio = np.zeros(samples)
        
        # Add simulated speaker signals
        for i in range(num_speakers):
            # Different frequency for each speaker
            freq = 200 + i * 100
            speaker_signal = 0.3 * np.sin(2 * np.pi * freq * np.arange(samples) / sample_rate)
            
            # Add some overlap
            start = int(i * samples / (num_speakers + 1))
            end = min(start + int(samples * 0.7), samples)
            audio[start:end] += speaker_signal[:end-start]
        
        # Add noise
        audio += 0.05 * np.random.randn(samples)
        
        return audio.astype(np.float32)
    
    async def benchmark_latency(self, audio_durations: List[float] = [0.5, 1.0, 2.0, 5.0]):
        """Benchmark processing latency for different audio lengths"""
        logger.info("Running latency benchmarks...")
        
        latency_results = []
        
        for duration in audio_durations:
            logger.info(f"Testing {duration}s audio...")
            
            # Generate test audio
            audio = self.generate_test_audio(duration)
            
            # Create audio segment
            segment = AudioSegment(
                data=audio,
                sample_rate=16000,
                timestamp=0.0,
                duration=duration
            )
            
            # Run multiple times for averaging
            latencies = []
            for i in range(5):
                start_time = time.time()
                result = await self.pipeline.process_audio_segment(segment)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            latency_results.append({
                'audio_duration_s': duration,
                'avg_latency_ms': avg_latency,
                'std_latency_ms': std_latency,
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'real_time_factor': avg_latency / (duration * 1000)
            })
            
            logger.info(f"  Average latency: {avg_latency:.1f}ms (RTF: {avg_latency/(duration*1000):.3f})")
        
        self.results['benchmarks']['latency'] = latency_results
        return latency_results
    
    def benchmark_error_correction(self):
        """Benchmark NLP error correction accuracy"""
        logger.info("Running error correction benchmarks...")
        
        # Test cases with known corrections
        test_cases = [
            {
                "raw": "I think we sh## meet tomor- at thre-",
                "expected_keywords": ["think", "meet", "tomorrow", "three"],
                "severity": "moderate"
            },
            {
                "raw": "The budg## is appro### five thous###",
                "expected_keywords": ["budget", "five"],
                "severity": "high"
            },
            {
                "raw": "Can you he## me? The connec### is b##",
                "expected_keywords": ["hear", "connection"],
                "severity": "high"
            },
            {
                "raw": "Let's schedule the meeting for next week",
                "expected_keywords": ["schedule", "meeting", "next", "week"],
                "severity": "none"
            }
        ]
        
        correction_results = []
        
        for test_case in test_cases:
            # Create segment
            segment = TranscriptionSegment(
                speaker_id=1,
                start_time=0.0,
                end_time=3.0,
                raw_text=test_case["raw"]
            )
            
            # Time the correction
            start_time = time.time()
            if not hasattr(self, 'strategies'):
                self.strategies = CorrectionStrategies()
            segment.corrected_text = self.strategies.pattern_based_correction(segment.raw_text)
            correction_time = (time.time() - start_time) * 1000
            
            # Check accuracy
            corrected_lower = corrected_segment.corrected_text.lower()
            keywords_found = sum(1 for keyword in test_case["expected_keywords"] 
                               if keyword in corrected_lower)
            accuracy = keywords_found / len(test_case["expected_keywords"])
            
            correction_results.append({
                'raw_text': test_case["raw"],
                'corrected_text': corrected_segment.corrected_text,
                'severity': test_case["severity"],
                'accuracy': accuracy,
                'correction_time_ms': correction_time,
                'confidence': corrected_segment.confidence
            })
            
            logger.info(f"  {test_case['severity']} severity: {accuracy*100:.0f}% accuracy in {correction_time:.1f}ms")
        
        self.results['benchmarks']['error_correction'] = correction_results
        return correction_results
    
    async def benchmark_concurrent_speakers(self, max_speakers: int = 5):
        """Benchmark performance with multiple concurrent speakers"""
        logger.info("Running concurrent speaker benchmarks...")
        
        concurrent_results = []
        
        for num_speakers in range(1, max_speakers + 1):
            logger.info(f"Testing with {num_speakers} speakers...")
            
            # Generate audio with multiple speakers
            audio = self.generate_test_audio(2.0, num_speakers)
            
            segment = AudioSegment(
                data=audio,
                sample_rate=16000,
                timestamp=0.0,
                duration=2.0
            )
            
            # Process
            start_time = time.time()
            result = await self.pipeline.process_audio_segment(segment)
            processing_time = (time.time() - start_time) * 1000
            
            concurrent_results.append({
                'num_speakers': num_speakers,
                'processing_time_ms': processing_time,
                'segments_detected': len(result.segments),
                'stage_latencies': result.stage_latencies
            })
            
            logger.info(f"  Processing time: {processing_time:.1f}ms, Segments: {len(result.segments)}")
        
        self.results['benchmarks']['concurrent_speakers'] = concurrent_results
        return concurrent_results
    
    def benchmark_cache_performance(self):
        """Benchmark cache hit rates and performance improvement"""
        logger.info("Running cache performance benchmarks...")
        
        # Clear cache first
        self.pipeline.realtime_corrector.clear_cache()
        
        test_texts = [
            "I think we sh## meet tomorrow",
            "The budg## is ready",
            "Can you he## me clearly",
            "Let's disc### this later"
        ] * 5  # Repeat to test cache
        
        cache_results = []
        
        for i, text in enumerate(test_texts):
            result = self.pipeline.realtime_corrector.correct_with_deadline(text)
            
            cache_results.append({
                'iteration': i,
                'text': text[:20] + "...",  # Truncate for display
                'strategy': result.strategy,
                'latency_ms': result.latency_ms,
                'cache_hit': result.strategy == 'cache'
            })
        
        # Calculate cache statistics
        total_corrections = len(cache_results)
        cache_hits = sum(1 for r in cache_results if r['cache_hit'])
        cache_hit_rate = cache_hits / total_corrections
        
        avg_cache_latency = np.mean([r['latency_ms'] for r in cache_results if r['cache_hit']])
        avg_compute_latency = np.mean([r['latency_ms'] for r in cache_results if not r['cache_hit']])
        
        self.results['benchmarks']['cache_performance'] = {
            'total_corrections': total_corrections,
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'avg_cache_latency_ms': avg_cache_latency if cache_hits > 0 else 0,
            'avg_compute_latency_ms': avg_compute_latency,
            'speedup_factor': avg_compute_latency / avg_cache_latency if cache_hits > 0 else 1
        }
        
        logger.info(f"  Cache hit rate: {cache_hit_rate*100:.1f}%")
        logger.info(f"  Cache speedup: {avg_compute_latency/avg_cache_latency:.1f}x" if cache_hits > 0 else "  No cache hits")
        
        return self.results['benchmarks']['cache_performance']
    
    def generate_plots(self):
        """Generate visualization plots for benchmark results"""
        logger.info("Generating benchmark plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Latency vs Audio Duration
        if 'latency' in self.results['benchmarks']:
            plt.figure(figsize=(10, 6))
            data = self.results['benchmarks']['latency']
            durations = [d['audio_duration_s'] for d in data]
            latencies = [d['avg_latency_ms'] for d in data]
            rtf = [d['real_time_factor'] for d in data]
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(durations, latencies, alpha=0.7, label='Latency')
            ax1.set_xlabel('Audio Duration (seconds)')
            ax1.set_ylabel('Processing Latency (ms)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(durations, rtf, 'r-o', label='RTF')
            ax2.set_ylabel('Real-Time Factor', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            
            plt.title('Processing Latency vs Audio Duration')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'latency_benchmark.png')
            plt.close()
        
        # 2. Error Correction Accuracy
        if 'error_correction' in self.results['benchmarks']:
            plt.figure(figsize=(10, 6))
            data = self.results['benchmarks']['error_correction']
            severities = [d['severity'] for d in data]
            accuracies = [d['accuracy'] * 100 for d in data]
            times = [d['correction_time_ms'] for d in data]
            
            x = np.arange(len(severities))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(x - width/2, accuracies, width, label='Accuracy (%)', alpha=0.7)
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_xlabel('Error Severity')
            ax1.set_xticks(x)
            ax1.set_xticklabels(severities)
            
            ax2 = ax1.twinx()
            ax2.bar(x + width/2, times, width, label='Time (ms)', alpha=0.7, color='orange')
            ax2.set_ylabel('Correction Time (ms)')
            
            plt.title('Error Correction Performance by Severity')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'error_correction_benchmark.png')
            plt.close()
        
        # 3. Concurrent Speakers Performance
        if 'concurrent_speakers' in self.results['benchmarks']:
            plt.figure(figsize=(10, 6))
            data = self.results['benchmarks']['concurrent_speakers']
            speakers = [d['num_speakers'] for d in data]
            times = [d['processing_time_ms'] for d in data]
            
            plt.plot(speakers, times, 'b-o', linewidth=2, markersize=8)
            plt.xlabel('Number of Concurrent Speakers')
            plt.ylabel('Processing Time (ms)')
            plt.title('Processing Time vs Number of Speakers')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'concurrent_speakers_benchmark.png')
            plt.close()
        
        logger.info(f"Plots saved to {self.output_dir}")
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        logger.info("Generating benchmark report...")
        
        report_path = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save raw results
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        summary_path = self.output_dir / "benchmark_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Multi-Speaker ASR Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # System info
            f.write("System Information:\n")
            for key, value in self.results['system_info'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Latency summary
            if 'latency' in self.results['benchmarks']:
                f.write("Latency Performance:\n")
                for item in self.results['benchmarks']['latency']:
                    f.write(f"  {item['audio_duration_s']}s audio: "
                           f"{item['avg_latency_ms']:.1f}ms (RTF: {item['real_time_factor']:.3f})\n")
                f.write("\n")
            
            # Error correction summary
            if 'error_correction' in self.results['benchmarks']:
                f.write("Error Correction Accuracy:\n")
                for item in self.results['benchmarks']['error_correction']:
                    f.write(f"  {item['severity']} severity: "
                           f"{item['accuracy']*100:.0f}% accuracy\n")
                f.write("\n")
            
            # Cache performance
            if 'cache_performance' in self.results['benchmarks']:
                cache = self.results['benchmarks']['cache_performance']
                f.write("Cache Performance:\n")
                f.write(f"  Hit rate: {cache['cache_hit_rate']*100:.1f}%\n")
                f.write(f"  Speedup: {cache['speedup_factor']:.1f}x\n")
        
        logger.info(f"Report saved to {report_path}")
        logger.info(f"Summary saved to {summary_path}")


async def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="Benchmark Multi-Speaker ASR System")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Directory for benchmark results")
    parser.add_argument("--quick", action="store_true",
                      help="Run quick benchmarks only")
    parser.add_argument("--no-plots", action="store_true",
                      help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = ASRBenchmark(args.output_dir)
    
    try:
        # Run benchmarks
        if args.quick:
            # Quick benchmarks
            await benchmark.benchmark_latency([0.5, 1.0])
            benchmark.benchmark_error_correction()
        else:
            # Full benchmarks
            await benchmark.benchmark_latency()
            benchmark.benchmark_error_correction()
            await benchmark.benchmark_concurrent_speakers()
            benchmark.benchmark_cache_performance()
        
        # Generate visualizations
        if not args.no_plots:
            benchmark.generate_plots()
        
        # Generate report
        benchmark.generate_report()
        
        logger.info("\n✅ Benchmarking complete!")
        logger.info(f"Results saved to: {benchmark.output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
    finally:
        # Cleanup
        benchmark.pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())