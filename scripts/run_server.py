#!/usr/bin/env python3
"""
WebSocket server for real-time multi-speaker ASR with NLP enhancement.
Accepts audio streams and returns corrected transcriptions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import websockets
import json
import numpy as np
import base64
import yaml
from typing import Dict, Optional
from loguru import logger
from datetime import datetime

from src.pipeline.orchestrator import MultiSpeakerASRPipeline, AudioSegment

class ASRWebSocketServer:
    """WebSocket server for real-time ASR processing"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the server with configuration"""
        self.config = self._load_config(config_path)
        self.pipeline = MultiSpeakerASRPipeline(config_path)
        self.active_connections = set()
        self.processing_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'processed_segments': 0,
            'total_processing_time': 0
        }
        
        logger.info("ASR WebSocket Server initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load server configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('server', {})
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {
                'host': 'localhost',
                'port': 8765,
                'max_connections': 10
            }
    
    async def handle_client(self, websocket, path):
        """Handle a client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New connection from {client_id}")
        
        # Check connection limit
        if len(self.active_connections) >= self.config.get('max_connections', 10):
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Server at maximum capacity'
            }))
            await websocket.close()
            return
        
        # Add to active connections
        self.active_connections.add(websocket)
        self.processing_stats['total_connections'] += 1
        self.processing_stats['active_connections'] = len(self.active_connections)
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Connected to ASR server',
                'capabilities': {
                    'max_latency_ms': 100,
                    'supports_streaming': True,
                    'languages': ['en']
                }
            }))
            
            # Handle messages
            async for message in websocket:
                await self.process_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up
            self.active_connections.discard(websocket)
            self.processing_stats['active_connections'] = len(self.active_connections)
    
    async def process_message(self, websocket, message):
        """Process incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'audio':
                await self.process_audio_message(websocket, data)
            elif message_type == 'config':
                await self.process_config_message(websocket, data)
            elif message_type == 'stats':
                await self.send_stats(websocket)
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON'
            }))
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Processing error: {str(e)}'
            }))
    
    async def process_audio_message(self, websocket, data):
        """Process audio data and return transcription"""
        try:
            # Decode audio data
            audio_base64 = data.get('audio')
            if not audio_base64:
                raise ValueError("No audio data provided")
            
            audio_bytes = base64.b64decode(audio_base64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Get metadata
            sample_rate = data.get('sample_rate', 16000)
            timestamp = data.get('timestamp', 0.0)
            
            # Create audio segment
            audio_segment = AudioSegment(
                data=audio_array,
                sample_rate=sample_rate,
                timestamp=timestamp,
                duration=len(audio_array) / sample_rate
            )
            
            # Process through pipeline
            start_time = datetime.now()
            result = await self.pipeline.process_audio_segment(audio_segment)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update stats
            self.processing_stats['processed_segments'] += len(result.segments)
            self.processing_stats['total_processing_time'] += processing_time
            
            # Send results
            response = {
                'type': 'transcription',
                'timestamp': timestamp,
                'processing_time_ms': processing_time,
                'segments': []
            }
            
            for segment in result.segments:
                response['segments'].append({
                    'speaker_id': segment.speaker_id,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'raw_text': segment.raw_text,
                    'corrected_text': segment.corrected_text,
                    'confidence': segment.confidence
                })
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Audio processing failed: {str(e)}'
            }))
    
    async def process_config_message(self, websocket, data):
        """Update configuration for this connection"""
        # In production, you might want to maintain per-connection configs
        config_update = data.get('config', {})
        
        # Validate and apply config updates
        valid_keys = ['language', 'max_speakers', 'enable_speaker_tracking']
        applied_updates = {}
        
        for key, value in config_update.items():
            if key in valid_keys:
                applied_updates[key] = value
        
        await websocket.send(json.dumps({
            'type': 'config_updated',
            'applied': applied_updates
        }))
    
    async def send_stats(self, websocket):
        """Send server statistics"""
        # Get pipeline stats
        pipeline_stats = self.pipeline.get_performance_metrics()
        
        stats = {
            'type': 'stats',
            'server': self.processing_stats,
            'pipeline': pipeline_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        await websocket.send(json.dumps(stats))
    
    async def broadcast_message(self, message: Dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            await asyncio.gather(
                *[ws.send(json.dumps(message)) for ws in self.active_connections],
                return_exceptions=True
            )
    
    async def start_server(self):
        """Start the WebSocket server"""
        host = self.config.get('host', 'localhost')
        port = self.config.get('port', 8765)
        
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async with websockets.serve(self.handle_client, host, port):
            logger.info(f"Server listening on ws://{host}:{port}")
            
            # Keep server running
            await asyncio.Future()  # Run forever

# Example client code (for reference)
EXAMPLE_CLIENT_CODE = """
// JavaScript WebSocket client example

const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
    console.log('Connected to ASR server');
    
    // Send audio data
    const audioData = new Float32Array(16000); // 1 second of audio
    const base64Audio = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
    
    ws.send(JSON.stringify({
        type: 'audio',
        audio: base64Audio,
        sample_rate: 16000,
        timestamp: Date.now() / 1000
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'transcription') {
        data.segments.forEach(segment => {
            console.log(`Speaker ${segment.speaker_id}: ${segment.corrected_text}`);
        });
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
"""

def main():
    """Main function to start the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASR WebSocket Server')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--host', type=str, help='Override host from config')
    parser.add_argument('--port', type=int, help='Override port from config')
    
    args = parser.parse_args()
    
    # Create and configure server
    server = ASRWebSocketServer(args.config)
    
    # Override config if specified
    if args.host:
        server.config['host'] = args.host
    if args.port:
        server.config['port'] = args.port
    
    # Print example client code
    print("\n" + "="*60)
    print("Server starting...")
    print("Example client code:")
    print("="*60)
    print(EXAMPLE_CLIENT_CODE)
    print("="*60 + "\n")
    
    # Run server
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()