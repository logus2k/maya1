#!/usr/bin/env python3

"""
Simple command-line test client for Maya-1-Voice TTS Server

Usage:
    python test_maya_client.py "Hello, this is a test!"
    python test_maya_client.py "Hello!" --voice af_bella --output test.wav
    python test_maya_client.py --interactive
"""

import asyncio
import argparse
import base64
import sys
import time
import wave
from pathlib import Path
import socketio

# Configuration
DEFAULT_SERVER = "http://localhost:7766"
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

class TTSClient:
    """Simple TTS client for testing."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.sio = socketio.AsyncClient()
        self.audio_chunks = []
        self.generation_complete = False
        self.error_message = None
        self.client_id = f"test-client-{int(time.time())}"
        
        # Register event handlers
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('audio_connection_registered', self.on_audio_registered)
        self.sio.on('tts_start', self.on_tts_start)
        self.sio.on('tts_audio', self.on_tts_audio)
        self.sio.on('tts_complete', self.on_tts_complete)
        self.sio.on('tts_error', self.on_tts_error)
    
    async def on_connect(self):
        print(f"✅ Connected to server: {self.server_url}")
        # Register audio connection
        await self.sio.emit('register_audio_connection', {
            'client_id': self.client_id
        })
    
    async def on_disconnect(self):
        print(f"🔌 Disconnected from server")
    
    async def on_audio_registered(self, data):
        print(f"🎧 Audio connection registered: {data.get('client_id')}")
    
    async def on_tts_start(self, data):
        print(f"🎤 TTS generation started")
        print(f"   Text: {data.get('text')}")
        print(f"   Voice: {data.get('voice')}")
        self.audio_chunks = []
        self.generation_complete = False
        self.error_message = None
    
    async def on_tts_audio(self, data):
        chunk_num = data.get('chunk', 0)
        audio_b64 = data.get('audio')
        
        if audio_b64:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)
            self.audio_chunks.append(audio_bytes)
            
            if chunk_num == 1:
                print(f"⚡ First chunk received ({len(audio_bytes)} bytes)")
            elif chunk_num % 10 == 0:
                print(f"📦 Chunk {chunk_num} received...")
    
    async def on_tts_complete(self, data):
        chunks = data.get('chunks', 0)
        duration = data.get('duration', 0)
        print(f"✅ Generation complete!")
        print(f"   Total chunks: {chunks}")
        print(f"   Duration: {duration:.2f}s")
        self.generation_complete = True
    
    async def on_tts_error(self, data):
        error = data.get('error', 'Unknown error')
        print(f"❌ Error: {error}")
        self.error_message = error
        self.generation_complete = True
    
    async def connect(self):
        """Connect to the server."""
        print(f"🔌 Connecting to {self.server_url}...")
        await self.sio.connect(self.server_url)
        # Wait for registration
        await asyncio.sleep(0.5)
    
    async def disconnect(self):
        """Disconnect from the server."""
        await self.sio.disconnect()
    
    async def generate_speech(self, text: str, voice: str = DEFAULT_VOICE) -> bytes:
        """
        Generate speech and return audio bytes.
        
        Args:
            text: Text to synthesize
            voice: Voice ID to use
            
        Returns:
            Audio bytes (PCM int16, 24kHz mono)
        """
        if not self.sio.connected:
            raise RuntimeError("Not connected to server")
        
        print(f"\n🎵 Generating speech...")
        print(f"   Text: '{text}'")
        print(f"   Voice: {voice}")
        
        # Reset state
        self.audio_chunks = []
        self.generation_complete = False
        self.error_message = None
        
        # Send generation request
        start_time = time.time()
        await self.sio.emit('generate_speech', {
            'client_id': self.client_id,
            'text': text,
            'voice': voice,
            'mode': 'tts'
        })
        
        # Wait for completion
        timeout = 60  # 60 second timeout
        elapsed = 0
        while not self.generation_complete and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed = time.time() - start_time
        
        if not self.generation_complete:
            raise TimeoutError(f"Generation timed out after {timeout}s")
        
        if self.error_message:
            raise RuntimeError(f"Generation error: {self.error_message}")
        
        # Combine all audio chunks
        total_audio = b''.join(self.audio_chunks)
        
        if not total_audio:
            raise RuntimeError("No audio generated")
        
        total_time = time.time() - start_time
        audio_duration = len(total_audio) / 2 / SAMPLE_RATE  # int16 = 2 bytes
        print(f"\n📊 Statistics:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Audio duration: {audio_duration:.2f}s")
        print(f"   Total bytes: {len(total_audio)}")
        print(f"   Chunks received: {len(self.audio_chunks)}")
        
        return total_audio
    
    def save_wav(self, audio_bytes: bytes, output_path: str):
        """Save audio bytes to WAV file."""
        with wave.open(output_path, 'wb') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(SAMPLE_RATE)  # 24kHz
            wav.writeframes(audio_bytes)
        
        print(f"💾 Saved to: {output_path}")


async def test_single(text: str, voice: str, output: str, server: str):
    """Test single TTS generation."""
    client = TTSClient(server)
    
    try:
        # Connect
        await client.connect()
        
        # Generate speech
        audio = await client.generate_speech(text, voice)
        
        # Save to file
        if output:
            client.save_wav(audio, output)
        else:
            # Auto-generate filename
            safe_text = "".join(c if c.isalnum() else "_" for c in text[:30])
            output_path = f"output_{safe_text}_{voice}.wav"
            client.save_wav(audio, output_path)
        
    finally:
        await client.disconnect()


async def test_interactive(server: str):
    """Interactive testing mode."""
    client = TTSClient(server)
    
    try:
        await client.connect()
        
        print("\n" + "="*60)
        print("🎤 Interactive TTS Test Mode")
        print("="*60)
        print("\nCommands:")
        print("  Type text to generate speech")
        print("  'voice <voice_id>' - Change voice")
        print("  'voices' - List available voices")
        print("  'quit' - Exit")
        print()
        
        current_voice = DEFAULT_VOICE
        
        while True:
            try:
                user_input = input(f"[{current_voice}] > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'voices':
                    print("\n📋 Available voices:")
                    voices = [
                        "af_heart (Warm female, American)",
                        "af_bella (Energetic female, American)",
                        "af_nicole (Professional female, American)",
                        "af_sarah (Mature female, American)",
                        "am_michael (Neutral male, American)",
                        "am_adam (Deep male, American)",
                        "bf_emma (Refined female, British)",
                        "bf_alice (Cheerful female, British)",
                        "bm_george (Professional male, British)",
                        "bm_lewis (Friendly male, British)",
                    ]
                    for v in voices:
                        print(f"  • {v}")
                    print()
                    continue
                
                if user_input.lower().startswith('voice '):
                    new_voice = user_input[6:].strip()
                    current_voice = new_voice
                    print(f"✅ Voice changed to: {current_voice}\n")
                    continue
                
                # Generate speech
                audio = await client.generate_speech(user_input, current_voice)
                
                # Save with timestamp
                timestamp = int(time.time())
                output_path = f"test_{timestamp}_{current_voice}.wav"
                client.save_wav(audio, output_path)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    finally:
        await client.disconnect()


async def test_multiple(server: str):
    """Test multiple voices with the same text."""
    client = TTSClient(server)
    
    test_text = "Hello! This is a test of the Maya-1-Voice text-to-speech system."
    test_voices = ['af_heart', 'am_michael', 'bf_emma', 'bm_george']
    
    try:
        await client.connect()
        
        print("\n" + "="*60)
        print("🎤 Testing Multiple Voices")
        print("="*60)
        print(f"\nText: '{test_text}'")
        print(f"Voices: {', '.join(test_voices)}\n")
        
        for voice in test_voices:
            print(f"\n{'='*60}")
            print(f"Testing voice: {voice}")
            print(f"{'='*60}")
            
            audio = await client.generate_speech(test_text, voice)
            output_path = f"test_multi_{voice}.wav"
            client.save_wav(audio, output_path)
            
            # Short pause between generations
            await asyncio.sleep(1)
        
        print(f"\n✅ All tests complete!")
        
    finally:
        await client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description='Test client for Maya-1-Voice TTS Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single generation
  python test_maya_client.py "Hello world!"
  
  # With specific voice and output
  python test_maya_client.py "Hello!" --voice bf_emma --output test.wav
  
  # Interactive mode
  python test_maya_client.py --interactive
  
  # Test multiple voices
  python test_maya_client.py --multi
  
  # Custom server
  python test_maya_client.py "Hello!" --server http://192.168.1.100:7766
        """
    )
    
    parser.add_argument('text', nargs='?', help='Text to synthesize')
    parser.add_argument('--voice', '-v', default=DEFAULT_VOICE,
                       help=f'Voice ID (default: {DEFAULT_VOICE})')
    parser.add_argument('--output', '-o', help='Output WAV file path')
    parser.add_argument('--server', '-s', default=DEFAULT_SERVER,
                       help=f'Server URL (default: {DEFAULT_SERVER})')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--multi', '-m', action='store_true',
                       help='Test multiple voices')
    
    args = parser.parse_args()
    
    # Validate
    if not args.interactive and not args.multi and not args.text:
        parser.error("Text is required unless using --interactive or --multi")
    
    # Run appropriate test
    try:
        if args.interactive:
            asyncio.run(test_interactive(args.server))
        elif args.multi:
            asyncio.run(test_multiple(args.server))
        else:
            asyncio.run(test_single(args.text, args.voice, args.output, args.server))
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
