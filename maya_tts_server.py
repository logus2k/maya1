#!/usr/bin/env python3

"""
Maya-1-Voice TTS Server
Compatible with existing Kokoro TTS client interface

Requirements:
pip install vllm transformers torch snac numpy fastapi uvicorn python-socketio soundfile
"""

import asyncio
import base64
import io
import logging
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime
import time
import numpy as np
import soundfile as sf
from fastapi import FastAPI
import socketio
import uvicorn
from contextlib import asynccontextmanager
import json
from pathlib import Path
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from snac import SNAC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MAYA-1-VOICE CONSTANTS
# ============================================================================

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_SAMPLE_RATE = 24000
SNAC_TOKENS_PER_FRAME = 7

DEFAULT_TEMPERATURE = 0.3  # Lower for more consistent generation
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 3000  # Reduced to prevent runaway generation
DEFAULT_MIN_TOKENS = 35  # Ensure at least 5 frames (5*7=35)
DEFAULT_REPETITION_PENALTY = 1.2  # Higher to prevent loops

# ============================================================================
# VOICE MAPPING (Kokoro voice IDs -> Maya-1 descriptions)
# ============================================================================

VOICE_DESCRIPTION_MAP = {
    # American English Female
    'af_heart': {
        'description': 'Female in their 30s with warm, friendly and conversational voice.',
        'temperature': 0.3,  # Lower for consistency
        'lang': 'en'
    },
    'af_bella': {
        'description': 'Female in their 20s with bright, energetic and upbeat voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    'af_nicole': {
        'description': 'Female in their 30s with clear, professional and neutral voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    'af_sarah': {
        'description': 'Female in their 40s with rich, calm and mature voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    
    # American English Male
    'am_michael': {
        'description': 'Male in their 30s with warm, conversational and neutral voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    'am_adam': {
        'description': 'Male in their 40s with deep, authoritative and confident voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    
    # British English Female
    'bf_emma': {
        'description': 'Female in their 30s with british accent and refined, polite voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    'bf_alice': {
        'description': 'Female in their 20s with british accent and cheerful, light voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    
    # British English Male
    'bm_george': {
        'description': 'Male in their 40s with british accent and deep, professional voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
    'bm_lewis': {
        'description': 'Male in their 30s with british accent and clear, friendly voice.',
        'temperature': 0.3,
        'lang': 'en'
    },
}

# Default fallback for unsupported voices
DEFAULT_VOICE_CONFIG = {
    'description': 'Realistic voice in their 30s with american accent. Neutral pitch, warm timbre, conversational pacing, neutral tone at medium intensity.',
    'temperature': 0.4,
    'lang': 'en'
}

# ============================================================================
# SNAC DECODER
# ============================================================================

class SNACDecoder:
    """Decodes SNAC tokens to audio waveforms."""
    
    def __init__(self, device: str = "cuda", model_path: str = "models/snac_24khz"):
        self.device = device
        logger.info(f"🎵 Loading SNAC 24kHz model from {model_path} to {device}...")
        self.snac_model = SNAC.from_pretrained(model_path, local_files_only=True).eval().to(device)
        logger.info(f"✅ SNAC decoder initialized")
    
    def unpack_snac_from_7(self, vocab_ids: List[int]) -> List[List[int]]:
        """Unpack 7-token SNAC frames to 3 hierarchical levels."""
        if vocab_ids and vocab_ids[-1] == CODE_END_TOKEN_ID:
            vocab_ids = vocab_ids[:-1]
        
        frames = len(vocab_ids) // SNAC_TOKENS_PER_FRAME
        vocab_ids = vocab_ids[:frames * SNAC_TOKENS_PER_FRAME]
        
        if frames == 0:
            return [[], [], []]
        
        l1, l2, l3 = [], [], []
        
        for i in range(frames):
            slots = vocab_ids[i*7:(i+1)*7]
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
            ])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        
        return [l1, l2, l3]
    
    @torch.inference_mode()
    def decode(self, snac_tokens: List[int], use_sliding_window: bool = False) -> Optional[np.ndarray]:
        """
        Decode SNAC tokens to audio waveform.
        
        Args:
            snac_tokens: List of SNAC token IDs (7*n tokens)
            use_sliding_window: If True, return only middle 2048 samples
        
        Returns:
            Audio waveform as float32 numpy array, 24kHz mono
        """
        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            return None
        
        levels = self.unpack_snac_from_7(snac_tokens)
        if not levels[0]:
            return None
        
        codes = [
            torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0)
            for level in levels
        ]
        
        z_q = self.snac_model.quantizer.from_codes(codes)
        audio = self.snac_model.decoder(z_q)
        audio = audio[0, 0].cpu().numpy()
        
        # Sliding window mode: keep middle 2048 samples only
        if use_sliding_window and len(audio) >= 4096:
            audio = audio[2048:4096]
        
        return audio
    
    def decode_to_bytes(self, snac_tokens: List[int], use_sliding_window: bool = False) -> Optional[bytes]:
        """
        Decode SNAC tokens to audio bytes (int16 PCM).
        
        Args:
            snac_tokens: List of SNAC token IDs
            use_sliding_window: Use sliding window for smooth streaming
        
        Returns:
            Audio as bytes (int16 PCM, 24kHz mono)
        """
        audio = self.decode(snac_tokens, use_sliding_window=use_sliding_window)
        if audio is None:
            return None
        
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

# ============================================================================
# MAYA-1-VOICE MODEL
# ============================================================================

class Maya1VoiceModel:
    """Maya-1-Voice VLLM model wrapper."""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
    ):
        self.model_path = model_path
        logger.info(f"🚀 Initializing Maya-1-Voice Model")
        logger.info(f"📁 Model: {model_path}")
        logger.info(f"🔢 Dtype: {dtype}")
        
        # Load tokenizer
        logger.info(f"📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        logger.info(f"✅ Tokenizer loaded: {len(self.tokenizer)} tokens")
        
        # Initialize VLLM engine
        logger.info(f"🔧 Initializing VLLM engine...")
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,
            max_num_seqs=1,  # Process one request at a time to save memory
            enable_prefix_caching=False,  # Disable prefix caching to save memory
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"✅ VLLM engine ready")
    
    def build_prompt(self, description: str, text: str) -> str:
        """
        Build Maya-1-Voice prompt with explicit special tokens.
        
        This matches the official Maya1 format:
        SOH + BOS + formatted_text + EOT + EOH + SOA + SOS
        
        Args:
            description: Voice description
            text: Text to synthesize (with optional emotion tags like <laugh_harder>)
            
        Returns:
            Formatted prompt string
        """
        # Special token IDs (matching your working example)
        SOH_ID = 128259
        EOH_ID = 128260
        SOA_ID = 128261
        TEXT_EOT_ID = 128009
        
        # Decode special tokens
        soh_token = self.tokenizer.decode([SOH_ID])
        eoh_token = self.tokenizer.decode([EOH_ID])
        soa_token = self.tokenizer.decode([SOA_ID])
        sos_token = self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token = self.tokenizer.decode([TEXT_EOT_ID])
        bos_token = self.tokenizer.bos_token
        
        # Format: <description="..."> text
        formatted_text = f'<description="{description}"> {text}'
        
        # Build prompt with explicit token sequence
        prompt = (
            soh_token + bos_token + formatted_text + eot_token +
            eoh_token + soa_token + sos_token
        )
        
        return prompt

# ============================================================================
# STREAMING PIPELINE
# ============================================================================

class Maya1VoiceStreamingPipeline:
    """Streaming TTS pipeline with chunked delivery."""
    
    def __init__(self, model: Maya1VoiceModel, snac_decoder: SNACDecoder):
        self.model = model
        self.snac_decoder = snac_decoder
        logger.info(f"🌊 Maya-1-Voice Streaming Pipeline initialized")
    
    async def generate_speech_stream(
        self,
        description: str,
        text: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech audio with sliding window streaming.
        
        Uses the same approach as the working stream.py example.
        """
        logger.info(f"🌊 Starting sliding window streaming generation")
        logger.debug(f"📝 Description: {description[:80]}...")
        logger.debug(f"💬 Text: {text}")
        
        # Build prompt
        prompt = self.model.build_prompt(description, text)
        
        # Configure sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=DEFAULT_MIN_TOKENS,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[CODE_END_TOKEN_ID],
        )
        
        # Token buffer
        token_buffer = []
        total_tokens = 0
        total_chunks = 0
        
        # Generate with VLLM
        import uuid
        request_id = f"maya1voice-{uuid.uuid4().hex[:8]}-{int(time.time() * 1000000)}"
        
        results_generator = self.model.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        
        # Stream tokens with sliding window decoding (matching stream.py)
        first_tokens_logged = False
        async for request_output in results_generator:
            generated_ids = request_output.outputs[0].token_ids
            
            # Process only new tokens
            new_tokens = generated_ids[total_tokens:]
            total_tokens = len(generated_ids)
            
            # Debug: Log first few tokens
            if not first_tokens_logged and len(new_tokens) > 0:
                logger.debug(f"🔍 First 10 generated tokens: {new_tokens[:10]}")
                first_tokens_logged = True
            
            # Filter and buffer SNAC tokens
            non_snac_count = 0
            for token_id in new_tokens:
                if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID:
                    token_buffer.append(token_id)
                elif token_id != CODE_END_TOKEN_ID:
                    # Non-SNAC token detected (text generation)
                    non_snac_count += 1
                    if non_snac_count <= 5:  # Log first few
                        logger.warning(f"⚠️  Non-SNAC token detected: {token_id}")
            
            if non_snac_count > 0:
                logger.warning(f"⚠️  Total non-SNAC tokens in batch: {non_snac_count} (model generating text!)")
            
            # Sliding window: process every 7 tokens when buffer > 27
            # Take last 28 tokens (4 frames) for smooth overlap
            if len(token_buffer) % 7 == 0 and len(token_buffer) > 27:
                window_tokens = token_buffer[-28:]
                
                # Decode with sliding window (returns middle 2048 samples)
                audio_bytes = self.snac_decoder.decode_to_bytes(
                    window_tokens,
                    use_sliding_window=True
                )
                
                if audio_bytes:
                    total_chunks += 1
                    if total_chunks == 1:
                        logger.info(f"⚡ First chunk decoded ({len(audio_bytes)} bytes)")
                    yield audio_bytes
        
        logger.info(f"✅ Streaming complete: {total_tokens} tokens → {total_chunks} chunks")

# ============================================================================
# GLOBAL STATE
# ============================================================================

maya_pipeline: Optional[Maya1VoiceStreamingPipeline] = None
tts_settings = None
client_tts_sessions: Dict[str, dict] = {}
audio_client_mapping: Dict[str, str] = {}

# ============================================================================
# INITIALIZATION
# ============================================================================

async def initialize_maya(model_path: str):
    """Initialize Maya-1-Voice pipeline."""
    global maya_pipeline
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 Using device: {device}")
        
        # Initialize model
        model = Maya1VoiceModel(
            model_path=model_path,
            dtype="bfloat16",
            max_model_len=4096,  # Reduced from 8192 to save VRAM
            gpu_memory_utilization=0.6,  # Reduced from 0.8 to leave headroom
        )
        
        # Initialize SNAC decoder (use local model like stream.py)
        snac_decoder = SNACDecoder(device=device, model_path="models/snac_24khz")
        
        # Create pipeline
        maya_pipeline = Maya1VoiceStreamingPipeline(model, snac_decoder)
        
        logger.info(f"✅ Maya-1-Voice pipeline ready")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Maya-1-Voice: {e}")
        raise

async def initialize_tts_settings(settings: dict):
    """Initialize TTS settings."""
    global tts_settings
    tts_settings = settings.get("TTS", {
        "default_voice": "af_heart",
        "default_speed": 1.0,  # Note: Speed not supported in Maya-1
        "chunk_size": 28,
        "temperature": 0.4,
        "max_tokens": 2000,
    })
    
    logger.info(f"⚙️ TTS Settings loaded:")
    logger.info(f"  Default voice: {tts_settings['default_voice']}")
    logger.info(f"  Chunk size: {tts_settings.get('chunk_size', 28)} tokens")
    logger.info(f"  Temperature: {tts_settings.get('temperature', 0.4)}")

def get_default_voice() -> str:
    return tts_settings.get('default_voice', 'af_heart') if tts_settings else 'af_heart'

def get_default_speed() -> float:
    return tts_settings.get('default_speed', 1.0) if tts_settings else 1.0

def is_voice_enabled(voice: str) -> bool:
    """Check if voice is supported (all English voices are supported)."""
    return voice in VOICE_DESCRIPTION_MAP

def get_voice_config(voice: str) -> dict:
    """Get voice configuration."""
    return VOICE_DESCRIPTION_MAP.get(voice, DEFAULT_VOICE_CONFIG)

# ============================================================================
# FASTAPI + SOCKET.IO SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("🚀 Starting Maya-1-Voice TTS Server...")
    settings = load_config()
    await initialize_tts_settings(settings)
    
    model_path = settings.get("model_path", "models/maya1")
    await initialize_maya(model_path)
    
    asyncio.create_task(cleanup_stale_buffers())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware for REST endpoints
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=False,
    engineio_logger=False
)

# ============================================================================
# SOCKET.IO EVENTS
# ============================================================================

@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info(f"🔌 Client connected: {sid}")
    await sio.emit('connection_established', {'sid': sid}, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"🔌 Client disconnected: {sid}")
    
    # Cleanup sessions
    if sid in client_tts_sessions:
        del client_tts_sessions[sid]
    
    # Cleanup audio mapping
    client_ids_to_remove = [cid for cid, asid in audio_client_mapping.items() if asid == sid]
    for client_id in client_ids_to_remove:
        del audio_client_mapping[client_id]

@sio.event
async def register_audio_connection(sid, data):
    """Register audio connection for a client."""
    client_id = data.get('client_id')
    if not client_id:
        logger.warning(f"⚠️ No client_id in register_audio_connection")
        return
    
    audio_client_mapping[client_id] = sid
    logger.info(f"🎧 Registered audio connection: {client_id} -> {sid}")
    
    await sio.emit('audio_connection_registered', {
        'client_id': client_id,
        'audio_sid': sid
    }, room=sid)

@sio.event
async def generate_speech(sid, data):
    """Generate speech from text - main TTS endpoint."""
    try:
        client_id = data.get('client_id', sid)
        text = data.get('text', '')
        voice = data.get('voice', get_default_voice())
        speed = data.get('speed', get_default_speed())
        mode = data.get('mode', 'tts')
        
        if not text:
            logger.warning(f"⚠️ Empty text received from {client_id}")
            await sio.emit('tts_error', {
                'error': 'Empty text',
                'client_id': client_id
            }, room=sid)
            return
        
        logger.info(f"🎤 Generate speech: client={client_id}, voice={voice}, mode={mode}, text_len={len(text)}")
        
        # Get audio connection
        audio_sid = audio_client_mapping.get(client_id, sid)
        
        # Initialize session
        if audio_sid not in client_tts_sessions:
            client_tts_sessions[audio_sid] = {
                'client_id': client_id,
                'mode': mode,
                'started': time.time()
            }
        
        # Get voice configuration
        voice_config = get_voice_config(voice)
        
        # Use custom description if provided, otherwise use default from config
        description = data.get('description', voice_config['description'])
        temperature = voice_config.get('temperature', tts_settings.get('temperature', 0.3))
        
        logger.debug(f"📝 Using description: {description[:80]}...")
        
        # Note: Speed parameter is not supported by Maya-1 (would need post-processing)
        if speed != 1.0:
            logger.debug(f"⚠️ Speed parameter {speed} ignored (not supported by Maya-1)")
        
        # Send start event
        await sio.emit('tts_start', {
            'client_id': client_id,
            'text': text,
            'voice': voice,
            'timestamp': datetime.now().isoformat()
        }, room=audio_sid)
        
        # Generate and stream audio
        chunk_count = 0
        start_time = time.time()
        
        try:
            async for audio_chunk in maya_pipeline.generate_speech_stream(
                description=description,
                text=text,
                temperature=temperature,
            ):
                chunk_count += 1
                
                # Encode audio to base64
                audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                # Send audio chunk
                await sio.emit('tts_audio', {
                    'client_id': client_id,
                    'audio': audio_base64,
                    'chunk': chunk_count,
                    'sample_rate': SNAC_SAMPLE_RATE,
                    'timestamp': datetime.now().isoformat()
                }, room=audio_sid)
                
                if chunk_count == 1:
                    latency = time.time() - start_time
                    logger.info(f"⚡ First chunk latency: {latency:.2f}s")
            
            # Send completion event
            total_time = time.time() - start_time
            await sio.emit('tts_complete', {
                'client_id': client_id,
                'chunks': chunk_count,
                'duration': total_time,
                'timestamp': datetime.now().isoformat()
            }, room=audio_sid)
            
            logger.info(f"✅ Generation complete: {chunk_count} chunks in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            await sio.emit('tts_error', {
                'client_id': client_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, room=audio_sid)
        
    except Exception as e:
        logger.error(f"❌ generate_speech error: {e}", exc_info=True)
        await sio.emit('tts_error', {
            'client_id': data.get('client_id', sid),
            'error': str(e)
        }, room=sid)

@sio.event
async def stop_generation(sid, data):
    """Handle stop of TTS generation."""
    client_id = data.get('client_id')
    reason = data.get('reason', 'unknown')
    
    logger.info(f"🚨 STOP GENERATION: {client_id} (reason: {reason})")
    
    audio_sid = audio_client_mapping.get(client_id, client_id)
    
    await sio.emit('tts_stop_immediate', {
        'client_id': client_id,
        'reason': reason,
        'timestamp': datetime.now().isoformat()
    }, room=audio_sid)
    
    logger.info(f"🚨 Stop generation completed for {client_id}")

@sio.event
async def stop_current_generation(sid, data):
    """Handle immediate stop of current TTS generation."""
    await stop_generation(sid, data)

# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0-maya",
        "model": "Maya-1-Voice",
        "pipeline_ready": maya_pipeline is not None,
        "active_clients": len(client_tts_sessions),
        "supported_voices": len(VOICE_DESCRIPTION_MAP),
        "default_voice": get_default_voice(),
        "sample_rate": SNAC_SAMPLE_RATE,
        "language": "English only",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/voices")
async def get_voices():
    """Get available voices."""
    voices = []
    for voice_id, config in VOICE_DESCRIPTION_MAP.items():
        voices.append({
            'voice': voice_id,
            'description': config['description'],
            'language': 'English',
            'language_code': config['lang'],
            'enabled': True,
            'available': True
        })
    
    return {
        "voices": voices,
        "total_voices": len(voices),
        "enabled_voices": len(voices),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/languages")
async def get_languages():
    """Get supported languages."""
    return {
        "languages": [{
            'code': 'en',
            'name': 'English',
            'available': True,
            'enabled': True,
            'voices': list(VOICE_DESCRIPTION_MAP.keys()),
            'voice_count': len(VOICE_DESCRIPTION_MAP)
        }],
        "total_languages": 1,
        "enabled_languages": 1,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/settings")
async def get_settings():
    """Get TTS settings."""
    return {
        "default_voice": get_default_voice(),
        "default_speed": get_default_speed(),
        "chunk_size": tts_settings.get('chunk_size', 28) if tts_settings else 28,
        "temperature": tts_settings.get('temperature', 0.4) if tts_settings else 0.4,
        "max_tokens": tts_settings.get('max_tokens', 2000) if tts_settings else 2000,
        "sample_rate": SNAC_SAMPLE_RATE,
        "note": "Speed parameter is not supported by Maya-1-Voice",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def cleanup_stale_buffers():
    """Clean up stale buffers periodically."""
    while True:
        try:
            await asyncio.sleep(60)
            # Could add cleanup logic here if needed
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def load_config(path: str = "maya_tts_config.json") -> dict:
    """Load configuration from file."""
    config_file = Path(path)
    if not config_file.exists():
        logger.warning(f"⚠️ Config file {path} not found. Using defaults.")
        return {
            "model_path": "models/maya1",
            "host": "0.0.0.0",
            "port": 7700,
            "log_level": "info"
        }
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return {}

# ============================================================================
# MAIN
# ============================================================================

sio_asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

async def main():
    settings = load_config()
    
    config = uvicorn.Config(
        sio_asgi_app,
        host=settings.get("host", "0.0.0.0"),
        port=settings.get("port", 7700),
        log_level=settings.get("log_level", "info"),
    )
    
    server = uvicorn.Server(config)
    logger.info("🌍 Starting Maya-1-Voice TTS Server...")
    logger.info(f"📡 Server: http://{config.host}:{config.port}")
    logger.info(f"🔌 Socket.IO: http://{config.host}:{config.port}/socket.io/")
    logger.info(f"💡 Health check: http://{config.host}:{config.port}/health")
    logger.info(f"🎤 Voices: http://{config.host}:{config.port}/voices")
    logger.info(f"⚙️ Settings: http://{config.host}:{config.port}/settings")
    
    await server.serve()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
