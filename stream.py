"""
Maya-1-Voice VLLM Streaming Inference - Standalone Reference Implementation

This is a complete, self-contained example for using Maya-1-Voice TTS model with VLLM and SNAC.
Demonstrates streaming audio generation with sliding window approach for smooth playback.

Requirements:
    pip install vllm transformers torch snac numpy

Usage:
    python vllm_streaming_inference.py

Author: Maya-1-Voice Team
License: MIT
"""

import torch
import numpy as np
import asyncio
from typing import List, Optional, AsyncGenerator
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from snac import SNAC


# ============================================================================
# CONSTANTS
# ============================================================================

# Special control tokens
CODE_START_TOKEN_ID = 128257  # Start of Speech (SOS)
CODE_END_TOKEN_ID = 128258    # End of Speech (EOS) - stop token for audio
CODE_TOKEN_OFFSET = 128266    # Start of SNAC codes

# SNAC token range (7 tokens per frame, 4096 codes per level)
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937  # 128266 + (7 * 4096) - 1

# SNAC configuration
SNAC_MODEL_NAME = "models/snac_24khz"
SNAC_SAMPLE_RATE = 24000
SNAC_TOKENS_PER_FRAME = 7

# Generation parameters
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2000
DEFAULT_MIN_TOKENS = 28  # At least 4 SNAC frames
DEFAULT_REPETITION_PENALTY = 1.1


# ============================================================================
# SNAC DECODER
# ============================================================================

class SNACDecoder:
    """
    Decodes SNAC tokens (7-token frames) to audio waveforms.
    
    The unpacking logic converts flat 7-token frames back to hierarchical
    3-level SNAC codes (matching the training preprocessing exactly).
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize SNAC decoder with 24kHz model."""
        self.device = device
        print(f"🎵 Loading SNAC 24kHz model to {device}...")
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME, local_files_only=True).eval().to(device)
        print(f"✅ SNAC decoder initialized")
    
    def unpack_snac_from_7(self, vocab_ids: List[int]) -> List[List[int]]:
        """
        Unpack 7-token SNAC frames to 3 hierarchical levels.
        
        This is the EXACT INVERSE of training preprocessing.
        
        Frame structure (7 tokens per frame):
        [slot0, slot1, slot2, slot3, slot4, slot5, slot6]
        
        Unpacking to [L1, L2, L3]:
        - slot0 → L1[i]       (coarse: 1x rate)
        - slot1 → L2[2*i]     (medium: 2x rate, even)
        - slot2 → L3[4*i+0]   (fine: 4x rate)
        - slot3 → L3[4*i+1]
        - slot4 → L2[2*i+1]   (medium: odd)
        - slot5 → L3[4*i+2]
        - slot6 → L3[4*i+3]
        
        Args:
            vocab_ids: List of SNAC token IDs (128266-156937), length divisible by 7
        
        Returns:
            [L1, L2, L3] where L1=n, L2=2n, L3=4n elements
        """
        # Remove EOS token if present
        if vocab_ids and vocab_ids[-1] == CODE_END_TOKEN_ID:
            vocab_ids = vocab_ids[:-1]
        
        # Ensure complete frames
        frames = len(vocab_ids) // SNAC_TOKENS_PER_FRAME
        vocab_ids = vocab_ids[:frames * SNAC_TOKENS_PER_FRAME]
        
        if frames == 0:
            return [[], [], []]
        
        l1, l2, l3 = [], [], []
        
        for i in range(frames):
            slots = vocab_ids[i*7:(i+1)*7]
            
            # Subtract offset and mod 4096 to get original SNAC codes
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,  # Even
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,  # Odd
            ])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        
        return [l1, l2, l3]
    
    @torch.inference_mode()
    def decode(
        self, 
        snac_tokens: List[int], 
        use_sliding_window: bool = False
    ) -> Optional[np.ndarray]:
        """
        Decode SNAC tokens to audio waveform.
        
        Args:
            snac_tokens: List of SNAC token IDs (7*n tokens)
            use_sliding_window: If True, return only middle 2048 samples 
                               (for smooth streaming without pops/clicks)
        
        Returns:
            Audio waveform as float32 numpy array, 24kHz mono
        """
        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            return None
        
        # Unpack to 3 hierarchical levels
        levels = self.unpack_snac_from_7(snac_tokens)
        
        if not levels[0]:
            return None
        
        # Convert to tensors
        codes = [
            torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0)
            for level in levels
        ]
        
        # Decode through SNAC quantizer + decoder
        z_q = self.snac_model.quantizer.from_codes(codes)
        audio = self.snac_model.decoder(z_q)
        
        # Extract audio: [batch, 1, samples] → [samples]
        audio = audio[0, 0].cpu().numpy()
        
        # Sliding window mode: keep middle 2048 samples only
        # This eliminates popping/cracking in streaming by overlapping windows
        if use_sliding_window and len(audio) >= 4096:
            audio = audio[2048:4096]
        
        return audio
    
    def decode_to_bytes(
        self, 
        snac_tokens: List[int], 
        use_sliding_window: bool = False
    ) -> Optional[bytes]:
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
        
        # Convert float32 to int16 PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()


# ============================================================================
# CUSTOM LOGITS PROCESSOR
# ============================================================================

class OnlyAudioAfterSOS:
    """
    Restricts vocabulary to SNAC codes + EOS after SOS token.
    
    This prevents the model from generating text tokens during audio phase,
    which would cause "hallucination" where the model repeats description text
    instead of generating proper audio codes.
    """
    
    def __init__(self):
        self._seen_sos = False
    
    def __call__(
        self,
        prompt_token_ids: List[int],
        generated_token_ids: List[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply constraint: after SOS, only allow SNAC codes + EOS.
        
        Args:
            prompt_token_ids: Original prompt token IDs
            generated_token_ids: Tokens generated so far
            logits: Logits for next token [vocab_size]
        
        Returns:
            Modified logits with masked tokens
        """
        # Check if SOS has been generated
        if not self._seen_sos:
            all_token_ids = prompt_token_ids + generated_token_ids
            if CODE_START_TOKEN_ID in all_token_ids:
                self._seen_sos = True
            else:
                return logits  # No constraint yet
        
        # Apply constraint: mask all tokens except SNAC codes + EOS
        mask = torch.full_like(logits, float('-inf'))
        mask[SNAC_MIN_ID:SNAC_MAX_ID + 1] = 0  # Allow SNAC codes
        mask[CODE_END_TOKEN_ID] = 0            # Allow EOS
        
        return logits + mask
    
    def reset(self):
        """Reset state for reuse across generations."""
        self._seen_sos = False


# ============================================================================
# MAYA-1-VOICE MODEL
# ============================================================================

class Maya1VoiceModel:
    """
    Maya-1-Voice TTS Model with VLLM inference engine.
    
    Handles model loading, tokenizer initialization, and VLLM engine setup.
    """
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
    ):
        """
        Initialize Maya-1-Voice model with VLLM.
        
        Args:
            model_path: Path to model checkpoint (local or HuggingFace)
            dtype: Model precision (bfloat16 recommended)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory fraction to use (0.0-1.0)
        """
        self.model_path = model_path
        
        print(f"🚀 Initializing Maya-1-Voice Model")
        print(f"📁 Model: {model_path}")
        print(f"🔢 Dtype: {dtype}")
        
        # Load tokenizer (must be from checkpoint with emotion tags)
        print(f"📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print(f"✅ Tokenizer loaded: {len(self.tokenizer)} tokens")
        
        # Initialize VLLM async engine
        print(f"🔧 Initializing VLLM engine...")
        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print(f"✅ VLLM engine ready")
    
    def build_prompt(self, description: str, text: str) -> str:
        """
        Build prompt in Maya-1-Voice format using chat template.
        
        Format: Chat template with <description="..."> text as content
        
        The model expects:
        1. Description of voice/character
        2. Text to synthesize (optionally with <emotion> tags)
        
        Args:
            description: Voice description 
                Example: "Realistic male voice in the 30s age with american accent. 
                         Normal pitch, warm timbre, conversational pacing."
            text: Text to synthesize
                Example: "Hello world! <excited> This is amazing!"
        
        Returns:
            Formatted prompt string using chat template
        """
        content = f'<description="{description}"> {text}'
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ============================================================================
# STREAMING PIPELINE
# ============================================================================

class Maya1VoiceStreamingPipeline:
    """
    Streaming TTS pipeline using sliding window approach.
    
    This generates smooth audio by:
    1. Streaming tokens from VLLM as they're generated
    2. Every 7 tokens, decoding the last 28 tokens (4 frames) - sliding window
    3. Keeping only middle 2048 samples from each decode
    4. Creating natural overlap between chunks for artifact-free playback
    """
    
    def __init__(self, model: Maya1VoiceModel, snac_decoder: SNACDecoder):
        """Initialize streaming pipeline."""
        self.model = model
        self.snac_decoder = snac_decoder
        print(f"🌊 Maya-1-Voice Streaming Pipeline initialized")
    
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
        Generate speech audio with streaming.
        
        Args:
            description: Voice/character description
            text: Text to synthesize (with optional <emotion> tags)
            temperature: Sampling temperature (lower = more stable)
            top_p: Nucleus sampling
            max_tokens: Max SNAC tokens to generate
            repetition_penalty: Prevent repetition loops
        
        Yields:
            Audio chunks as bytes (int16 PCM, 24kHz mono)
        """
        print(f"\n🌊 Starting streaming generation")
        print(f"📝 Description: {description[:80]}...")
        print(f"💬 Text: {text}")
        
        # Build prompt
        prompt = self.model.build_prompt(description, text)
        
        # Configure sampling (removed custom logits processor for V1 compatibility)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=DEFAULT_MIN_TOKENS,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[CODE_END_TOKEN_ID],  # Stop on audio EOS
        )
        
        print(f"🎲 Sampling: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
        
        # Token buffer for sliding window
        token_buffer = []
        total_tokens = 0
        total_chunks = 0
        
        # Generate with VLLM
        import uuid
        import time
        request_id = f"maya1voice-{uuid.uuid4().hex[:8]}-{int(time.time() * 1000000)}"
        
        results_generator = self.model.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        
        # Stream tokens with sliding window decoding
        async for request_output in results_generator:

            generated_ids = request_output.outputs[0].token_ids

            # Process only new tokens
            new_tokens = generated_ids[total_tokens:]
            total_tokens = len(generated_ids)
            
            # Filter and buffer SNAC tokens only
            for token_id in new_tokens:
                if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID:
                    token_buffer.append(token_id)
                    
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
                                print(f"🎵 First chunk decoded ({len(audio_bytes)} bytes)")
                            yield audio_bytes
        
        print(f"✅ Streaming complete: {total_tokens} tokens → {total_chunks} chunks")


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    """
    Example usage of Maya-1-Voice streaming inference.
    
    This demonstrates:
    1. Model initialization
    2. SNAC decoder setup
    3. Streaming generation
    4. Audio chunk handling
    """
    
    # Configuration
    MODEL_PATH = "models/maya1"  # Local model path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("Maya-1-Voice VLLM Streaming Inference Example")
    print("=" * 80)
    
    # Initialize model
    model = Maya1VoiceModel(
        model_path=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.8,  # Reduced for available GPU memory (12GB free)
    )
    
    # Initialize SNAC decoder
    snac_decoder = SNACDecoder(device=DEVICE)
    
    # Create pipeline
    pipeline = Maya1VoiceStreamingPipeline(model, snac_decoder)
    
    # Example 1: Professional voice
    description = "Female in their 30s with a paused, calm and sexy voice."

    text = "Hello, my dear! Would you come along with me this evening? I'll be waiting for you..."
    
    print(f"\n{'='*80}")
    print("Example 1: Professional Voice")
    print(f"{'='*80}")
    
    audio_chunks = []
    async for chunk in pipeline.generate_speech_stream(
        description=description,
        text=text,
        temperature=0.4,
        max_tokens=500,
    ):
        audio_chunks.append(chunk)
        print(f"📦 Received chunk {len(audio_chunks)}: {len(chunk)} bytes")
    
    # Combine chunks
    full_audio = b''.join(audio_chunks)
    print(f"\n✅ Total audio: {len(full_audio)} bytes ({len(full_audio)//2} samples, {len(full_audio)/2/24000:.2f}s)")
    
    # Save audio (optional)
    try:
        import wave
        output_file = "output_example1.wav"
        with wave.open(output_file, 'wb') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(24000)  # 24kHz
            wav.writeframes(full_audio)
        print(f"💾 Saved to {output_file}")
    except ImportError:
        print(f"⚠️  Install 'wave' module to save audio files")
    
    # Example 2: Character voice with emotions
    print(f"\n{'='*80}")
    print("Example 2: Character Voice with Emotions")
    print(f"{'='*80}")
    
    description = (
        "Creative, dark_villain character. Male voice in their 40s with british accent. "
        "Low pitch, gravelly timbre, slow pacing, angry tone at high intensity."
    )
    text = "The darkness isn't coming... <angry> it's already here!"
    
    audio_chunks = []
    async for chunk in pipeline.generate_speech_stream(
        description=description,
        text=text,
        temperature=0.5,
        max_tokens=800,
    ):
        audio_chunks.append(chunk)
        print(f"📦 Received chunk {len(audio_chunks)}: {len(chunk)} bytes")
    
    full_audio = b''.join(audio_chunks)
    print(f"\n✅ Total audio: {len(full_audio)} bytes ({len(full_audio)//2} samples, {len(full_audio)/2/24000:.2f}s)")
    
    # Save audio
    try:
        import wave
        output_file = "output_example2.wav"
        with wave.open(output_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(full_audio)
        print(f"💾 Saved to {output_file}")
    except ImportError:
        pass
    
    print(f"\n{'='*80}")
    print("🎉 Examples complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
