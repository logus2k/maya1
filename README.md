# Maya1 TTS Server

Real-time streaming text-to-speech server using Maya1, the best open-source voice AI model with emotion support.

## Features

- **Real-time streaming** - Audio plays as it's generated (sub-100ms first chunk latency)
- **6 voice presets** - Custom characters with unique personalities and accents
- **Emotion support** - 20+ emotion tags (laugh, whisper, excited, etc.)
- **Unlimited text length** - Automatic chunking for texts of any size
- **Web interface** - Beautiful, responsive UI with live audio playback
- **WebSocket streaming** - Efficient real-time audio delivery via Socket.IO
- **Production-ready** - Stable voice generation, optimized VRAM usage

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on RTX 4090)
- Node.js or any lightweight Web server, e.g.: npx serve, nginx, Caddy.

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/logus2k/maya1
cd maya1
```

2. **Install Python dependencies**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers accelerate snac soundfile fastapi uvicorn python-socketio aiohttp vllm
```

3. **Download models**
```bash
# Maya-1-Voice model
huggingface-cli download maya-research/maya1 --local-dir models/maya1

# SNAC decoder
huggingface-cli download hubertsiuzdak/snac_24khz --local-dir models/snac_24khz
```

### Running the Server

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run server
python maya_tts_server.py
```

Server starts on `http://0.0.0.0:7766`

### Running the Web Client

**Option 1: Local file**
```bash
# Open directly in browser
explorer.exe index.html  # Windows/WSL
open index.html          # macOS
xdg-open index.html      # Linux
```

**Option 2: Development server**
```bash
npx serve -p 9632
# Open http://localhost:9632/index.html
```

## Usage

### Web Interface

1. Open `index.html` in your browser
2. Select a voice from the dropdown
3. Enter text (edit voice description and text to be spoken if you want to)
4. Click "Generate Speech" - audio plays immediately!

### Voice Presets

| Voice | Description |
|-------|-------------|
| **Jack** | Grizzled Caribbean pirate captain - deep, rumbling baritone seasoned by 40 years at sea |
| **Mary** | British call-center professional (30s) - supportive and clear |
| **James** | British professional reader (50s) - calm, mature, clear voice |
| **Mahika** | Indian accent professional (20s) - supportive and enthusiastic |
| **Rob** | American teenager (16s) - excited and energetic |
| **Sam** | Old West American (30s) - slow, strong, and relaxed |

### Emotion Tags

Use emotion tags inline in your text:

```
"Hello! <excited> This is amazing! <laugh> I can't believe it works so well!"
```

**Supported emotions:**
`angry`, `appalled`, `chuckle`, `cry`, `curious`, `disappointed`, `excited`, `exhale`, `gasp`, `giggle`, `gulp`, `laugh`, `laugh_harder`, `mischievous`, `sarcastic`, `scream`, `sigh`, `sing`, `snort`, `whisper`

### Custom Voice Descriptions

Edit the voice description field to create custom voices:

```
"Speak in the grizzled, weathered tones of an old Caribbean pirate captain."
"Female in their 20s with Indian accent, professional and supportive voice."
"Young men in their 16s with American accent, teenager and excited voice."
```

## API Reference

### Socket.IO Events

#### Client → Server

**`register_audio_connection`**
```javascript
socket.emit('register_audio_connection', {
  client_id: 'unique-client-id'
});
```

**`generate_speech`**
```javascript
socket.emit('generate_speech', {
  client_id: 'unique-client-id',
  text: 'Text to speak',
  voice: 'maya1_jack',
  description: 'Optional custom description',
  mode: 'tts'
});
```

#### Server → Client

**`tts_start`** - Generation started
```javascript
{
  client_id: string,
  text: string,
  voice: string,
  text_chunks: number,
  timestamp: string
}
```

**`tts_audio`** - Audio chunk (streaming)
```javascript
{
  client_id: string,
  audio: ArrayBuffer,   // Binary PCM int16 audio data
  chunk: number,
  sample_rate: 24000,
  timestamp: string
}
```

**`tts_complete`** - Generation finished
```javascript
{
  client_id: string,
  chunks: number,
  text_chunks: number,
  duration: number,
  timestamp: string
}
```

**`tts_error`** - Error occurred
```javascript
{
  error: string,
  client_id: string
}
```

### REST Endpoints

**`GET /health`** - Health check
```json
{
  "status": "healthy",
  "model": "maya-1-voice",
  "device": "cuda",
  "timestamp": "2025-12-06T12:00:00"
}
```

**`GET /voices`** - List available voices
```json
{
  "voices": ["maya1_jack", "maya1_mary", "maya1_james", "maya1_mahika", "maya1_Sam", "maya1_amara"],
  "default": "maya1_jack"
}
```

## Configuration

Edit `maya_tts_config.json`:

```json
{
  "model_path": "models/maya1",
  "host": "0.0.0.0",
  "port": 7766,
  "log_level": "info",
  "access_log": false,
  "TTS": {
    "default_voice": "maya1_jack",
    "default_speed": 1.0,
    "chunk_size": 28,
    "temperature": 0.3,
    "max_tokens": 3000
  }
}
```

### Key Settings

- **`temperature`**: Lower = more consistent (0.2-0.4 recommended)
- **`max_tokens`**: Maximum SNAC tokens per generation (3000 = ~7min audio)
- **`chunk_size`**: SNAC tokens per streaming chunk (28 = 4 frames)

## Architecture

### Server Stack

- **VLLM** - Fast LLM inference with async engine
- **FastAPI** - REST API endpoints
- **Socket.IO** - WebSocket communication
- **SNAC** - Audio decoder (24kHz mono PCM)
- **PyTorch** - Deep learning framework

### Audio Pipeline

```
Text Input
    ↓
Build Prompt (with special tokens)
    ↓
VLLM Generation (streaming SNAC tokens)
    ↓
Sliding Window Decoding (28 tokens → 4096 bytes)
    ↓
Socket.IO Emit (binary real-time chunks)
    ↓
Web Audio API (browser playback)
```

### Text Chunking

Long texts are automatically split at sentence boundaries:
- Max 500 chars per chunk
- Splits at `.`, `!`, `?` when possible
- Falls back to word boundaries if needed
- Processes chunks sequentially
- Seamless audio streaming across chunks

## Performance

**Typical Metrics:**
- First chunk latency: 1-2 seconds
- Streaming latency: ~100ms per chunk
- VRAM usage: 10-12GB (RTX 4090)
- Audio quality: 24kHz, 16-bit PCM mono
- Throughput: ~3-5 seconds audio/second generation

**Optimizations:**
- Sliding window decoding (reduces artifacts)
- Async streaming (non-blocking I/O)
- Efficient tokenization (explicit special tokens)
- Memory-optimized VLLM settings

## Licenses

This project uses:
- [Maya1](https://huggingface.co/maya-research/maya1) - Apache 2.0 license
- [SNAC](https://github.com/hubertsiuzdak/snac) - MIT license

## Credits

- **Maya-1-Voice** by Maya Research
- **SNAC** by Hubert Siuzdak
- **VLLM** for fast inference
- Built with FastAPI, Socket.IO, and Web Audio API

---
