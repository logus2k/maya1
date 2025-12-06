#!/usr/bin/env python3

"""
Simple command-line test client for Maya-1-Voice TTS Server

Usage:
    python test_maya_client.py "Hello, this is a test!"
    python test_maya_client.py "Hello!" --voice af_bella --output test.wav
    python test_maya_client.py --interactive
    python test_maya_client.py --interactive --play  # With real-time playback
"""

import asyncio
import argparse
import base64
import sys
import time
import wave
import os
from pathlib import Path
import socketio

# Configure PulseAudio for WSLg
os.environ['PULSE_SERVER'] = 'unix:/mnt/wslg/PulseServer'

# Try to import audio playback libraries
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not available. Install with: pip install pyaudio")
    print("   Audio will be saved to files only (no real-time playback)\n")

# Configuration
DEFAULT_SERVER = "http://localhost:7766"
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

class AudioPlayer:
    """Real-time audio player using PyAudio with WSL support."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, use_subprocess: bool = False):
        if not PYAUDIO_AVAILABLE and not use_subprocess:
            raise RuntimeError("PyAudio not available")
        
        self.sample_rate = sample_rate
        self.use_subprocess = use_subprocess
        self.pyaudio = None
        self.stream = None
        self.process = None
        
        if not use_subprocess:
            self.pyaudio = pyaudio.PyAudio()
            # Try to find a working output device
            self.device_index = self._find_output_device()
        
    def _find_output_device(self):
        """Find a working audio output device (WSL-friendly)."""
        if not self.pyaudio:
            return None
            
        info = self.pyaudio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        # Try to find PulseAudio device first (best for WSL)
        for i in range(num_devices):
            try:
                dev_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                if dev_info.get('maxOutputChannels') > 0:
                    name = dev_info.get('name', '')
                    if 'pulse' in name.lower() or 'default' in name.lower():
                        print(f"🔊 Using audio device: {name}")
                        return i
            except Exception:
                continue
        
        # Fall back to any output device
        for i in range(num_devices):
            try:
                dev_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                if dev_info.get('maxOutputChannels') > 0:
                    name = dev_info.get('name', '')
                    print(f"🔊 Using audio device: {name}")
                    return i
            except Exception:
                continue
        
        # No device found
        return None
        
    def start(self):
        """Start audio stream."""
        if self.use_subprocess:
            # Use aplay for WSL (more reliable)
            import subprocess
            try:
                self.process = subprocess.Popen(
                    ['aplay', '-f', 'S16_LE', '-r', str(self.sample_rate), '-c', '1'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"🔊 Using aplay for audio playback")
            except FileNotFoundError:
                raise RuntimeError("aplay not found. Install with: sudo apt install alsa-utils")
        else:
            if self.device_index is None:
                raise RuntimeError("No audio output device available")
            
            try:
                self.stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.device_index,
                    frames_per_buffer=1024
                )
            except Exception as e:
                # Try without specifying device
                try:
                    self.stream = self.pyaudio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        output=True,
                        frames_per_buffer=1024
                    )
                except Exception as e2:
                    raise RuntimeError(f"Failed to open audio stream: {e2}")
        
    def play_chunk(self, audio_bytes: bytes):
        """Play audio chunk immediately."""
        if self.use_subprocess and self.process:
            try:
                self.process.stdin.write(audio_bytes)
                self.process.stdin.flush()
            except Exception as e:
                print(f"⚠️  Subprocess playback error: {e}")
        elif self.stream:
            try:
                self.stream.write(audio_bytes)
            except Exception as e:
                print(f"⚠️  Audio playback error: {e}")
    
    def stop(self):
        """Stop and close audio stream."""
        if self.use_subprocess and self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=1)
            except Exception:
                pass
            self.process = None
        elif self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
    
    def close(self):
        """Cleanup PyAudio."""
        self.stop()
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception:
                pass

class TTSClient:
    """Simple TTS client for testing."""
    
    def __init__(self, server_url: str, enable_playback: bool = False, use_aplay: bool = False):
        self.server_url = server_url
        self.sio = socketio.AsyncClient()
        self.audio_chunks = []
        self.generation_complete = False
        self.error_message = None
        self.client_id = f"test-client-{int(time.time())}"
        self.enable_playback = enable_playback and (PYAUDIO_AVAILABLE or use_aplay)
        self.audio_player = None
        
        if self.enable_playback:
            self.audio_player = AudioPlayer(use_subprocess=use_aplay)
        
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
        if self.enable_playback:
            print(f"🎧 Audio connection registered (playback enabled)")
        else:
            print(f"🎧 Audio connection registered")
    
    async def on_tts_start(self, data):
        if self.enable_playback:
            print(f"🔊 Playing audio in real-time...")
            if self.audio_player:
                self.audio_player.start()
        else:
            print(f"🎤 Generating audio...")
        
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
            
            # Play immediately if enabled
            if self.enable_playback and self.audio_player:
                try:
                    self.audio_player.play_chunk(audio_bytes)
                except Exception as e:
                    print(f"⚠️  Playback error: {e}")
            
            if chunk_num == 1:
                print(f"⚡ First chunk received ({len(audio_bytes)} bytes)")
            elif chunk_num % 5 == 0:
                print(f"📦 Chunk {chunk_num}...")
    
    async def on_tts_complete(self, data):
        chunks = data.get('chunks', 0)
        duration = data.get('duration', 0)
        
        # Stop playback
        if self.enable_playback and self.audio_player:
            self.audio_player.stop()
        
        print(f"✅ Complete! ({chunks} chunks, {duration:.2f}s)")
        self.generation_complete = True
    
    async def on_tts_error(self, data):
        error = data.get('error', 'Unknown error')
        print(f"❌ Error: {error}")
        self.error_message = error
        self.generation_complete = True
        
        if self.enable_playback and self.audio_player:
            self.audio_player.stop()
    
    async def connect(self):
        """Connect to the server."""
        print(f"🔌 Connecting to {self.server_url}...")
        await self.sio.connect(self.server_url)
        await asyncio.sleep(0.5)
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.audio_player:
            self.audio_player.close()
        await self.sio.disconnect()
    
    async def generate_speech(self, text: str, voice: str = DEFAULT_VOICE, save_to_file: bool = True) -> bytes:
        """
        Generate speech and optionally return audio bytes.
        
        Args:
            text: Text to synthesize
            voice: Voice ID to use
            save_to_file: Whether to save to file
            
        Returns:
            Audio bytes (PCM int16, 24kHz mono)
        """
        if not self.sio.connected:
            raise RuntimeError("Not connected to server")
        
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
        timeout = 60
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
        audio_duration = len(total_audio) / 2 / SAMPLE_RATE
        
        if save_to_file:
            print(f"📊 Audio: {audio_duration:.2f}s, {len(self.audio_chunks)} chunks, {len(total_audio)} bytes")
        
        return total_audio
    
    def save_wav(self, audio_bytes: bytes, output_path: str):
        """Save audio bytes to WAV file."""
        with wave.open(output_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio_bytes)
        
        print(f"💾 Saved to: {output_path}")


async def test_single(text: str, voice: str, output: str, server: str, play: bool, use_aplay: bool):
    """Test single TTS generation."""
    client = TTSClient(server, enable_playback=play, use_aplay=use_aplay)
    
    try:
        await client.connect()
        audio = await client.generate_speech(text, voice, save_to_file=not play)
        
        # Save to file unless only playing
        if output or not play:
            if output:
                client.save_wav(audio, output)
            else:
                safe_text = "".join(c if c.isalnum() else "_" for c in text[:30])
                output_path = f"output_{safe_text}_{voice}.wav"
                client.save_wav(audio, output_path)
        
    finally:
        await client.disconnect()


async def test_interactive(server: str, play: bool, use_aplay: bool):
    """Interactive testing mode with continuous input."""
    client = TTSClient(server, enable_playback=play, use_aplay=use_aplay)
    
    try:
        await client.connect()
        
        print("\n" + "="*70)
        print("🎤 Interactive Maya-1-Voice TTS")
        print("="*70)
        
        if play and PYAUDIO_AVAILABLE:
            print("🔊 Real-time audio playback ENABLED")
        elif play and not PYAUDIO_AVAILABLE:
            print("⚠️  Audio playback disabled (PyAudio not available)")
            print("   Install with: pip install pyaudio")
        else:
            print("💾 Audio will be saved to files")
        
        print("\nCommands:")
        print("  Type text to generate speech")
        print("  'voice <voice_id>' - Change voice")
        print("  'voices' - List available voices")
        print("  'play on/off' - Toggle playback (if available)")
        print("  'quit' or Ctrl+C - Exit")
        print()
        
        current_voice = DEFAULT_VOICE
        playback_enabled = play and PYAUDIO_AVAILABLE
        save_to_disk = not playback_enabled
        
        while True:
            try:
                # Show prompt with current settings
                mode_indicator = "🔊" if playback_enabled else "💾"
                user_input = input(f"{mode_indicator} [{current_voice}] > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'voices':
                    print("\n📋 Available voices:")
                    voices = [
                        "af_heart (❤️ Warm female, American)",
                        "af_bella (🔥 Energetic female, American)",
                        "af_nicole (🎧 Professional female, American)",
                        "af_sarah (👩 Mature female, American)",
                        "am_michael (👨 Neutral male, American)",
                        "am_adam (🎙️ Deep male, American)",
                        "bf_emma (🇬🇧 Refined female, British)",
                        "bf_alice (😊 Cheerful female, British)",
                        "bm_george (🎩 Professional male, British)",
                        "bm_lewis (🙂 Friendly male, British)",
                    ]
                    for v in voices:
                        print(f"  • {v}")
                    print()
                    continue
                
                if user_input.lower().startswith('voice '):
                    new_voice = user_input[6:].strip()
                    current_voice = new_voice
                    print(f"✅ Voice: {current_voice}\n")
                    continue
                
                if user_input.lower() == 'play on':
                    if PYAUDIO_AVAILABLE:
                        playback_enabled = True
                        client.enable_playback = True
                        save_to_disk = False
                        print("✅ Real-time playback enabled\n")
                    else:
                        print("❌ PyAudio not available\n")
                    continue
                
                if user_input.lower() == 'play off':
                    playback_enabled = False
                    client.enable_playback = False
                    save_to_disk = True
                    print("✅ Playback disabled, saving to files\n")
                    continue
                
                # Generate speech
                audio = await client.generate_speech(user_input, current_voice, save_to_file=save_to_disk)
                
                # Save with timestamp if not playing
                if save_to_disk:
                    timestamp = int(time.time())
                    output_path = f"tts_{timestamp}_{current_voice}.wav"
                    client.save_wav(audio, output_path)
                elif playback_enabled:
                    # Save to temp and play immediately
                    import tempfile
                    import subprocess
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    client.save_wav(audio, tmp_path)
                    
                    # Try to play with available player
                    try:
                        # Try aplay first (WSL/Linux)
                        result = subprocess.run(
                            ['aplay', '-q', tmp_path],
                            capture_output=True,
                            timeout=10
                        )
                        if result.returncode != 0:
                            # Try paplay (PulseAudio)
                            subprocess.run(['paplay', tmp_path], timeout=10)
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        print("⚠️  Could not auto-play. File saved to:", tmp_path)
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                print()  # Blank line for readability
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    finally:
        await client.disconnect()


async def test_multiple(server: str, play: bool, use_aplay: bool):
    """Test multiple voices with the same text."""
    client = TTSClient(server, enable_playback=play, use_aplay=use_aplay)
    
    test_text = "Hello! This is a test of the Maya-1-Voice text-to-speech system."
    test_voices = ['af_heart', 'am_michael', 'bf_emma', 'bm_george']
    
    try:
        await client.connect()
        
        print("\n" + "="*70)
        print("🎤 Testing Multiple Voices")
        print("="*70)
        print(f"\nText: '{test_text}'")
        print(f"Voices: {', '.join(test_voices)}\n")
        
        for voice in test_voices:
            print(f"\n{'='*70}")
            print(f"Testing voice: {voice}")
            print(f"{'='*70}")
            
            audio = await client.generate_speech(test_text, voice, save_to_file=not play)
            
            if not play:
                output_path = f"test_multi_{voice}.wav"
                client.save_wav(audio, output_path)
            
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
  # Interactive mode with real-time playback
  python test_maya_client.py --interactive --play
  
  # Interactive mode (save to files)
  python test_maya_client.py --interactive
  
  # Single generation with playback
  python test_maya_client.py "Hello world!" --play
  
  # Single generation saved to file
  python test_maya_client.py "Hello!" --voice bf_emma --output test.wav
  
  # Test multiple voices
  python test_maya_client.py --multi
        """
    )
    
    parser.add_argument('text', nargs='?', help='Text to synthesize')
    parser.add_argument('--voice', '-v', default=DEFAULT_VOICE,
                       help=f'Voice ID (default: {DEFAULT_VOICE})')
    parser.add_argument('--output', '-o', help='Output WAV file path')
    parser.add_argument('--server', '-s', default=DEFAULT_SERVER,
                       help=f'Server URL (default: {DEFAULT_SERVER})')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode (continuous input)')
    parser.add_argument('--multi', '-m', action='store_true',
                       help='Test multiple voices')
    parser.add_argument('--play', '-p', action='store_true',
                       help='Enable real-time audio playback')
    parser.add_argument('--aplay', action='store_true',
                       help='Use aplay instead of PyAudio (better for WSL)')
    
    args = parser.parse_args()
    
    # Validate
    if not args.interactive and not args.multi and not args.text:
        parser.error("Text is required unless using --interactive or --multi")
    
    # Check PyAudio availability if playback requested
    if args.play and not PYAUDIO_AVAILABLE:
        print("⚠️  Warning: --play requires PyAudio")
        print("   Install with: pip install pyaudio")
        print("   Continuing without playback...\n")
    
    # Run appropriate test
    try:
        if args.interactive:
            asyncio.run(test_interactive(args.server, args.play, args.aplay))
        elif args.multi:
            asyncio.run(test_multiple(args.server, args.play, args.aplay))
        else:
            asyncio.run(test_single(args.text, args.voice, args.output, args.server, args.play, args.aplay))
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
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
