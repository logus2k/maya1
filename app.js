// Configuration
const SERVER_URL = 'http://localhost:7766';
const SAMPLE_RATE = 24000;

// Voice data (will be loaded from JSON)
let VOICE_DATA = null;
let VOICE_DESCRIPTIONS = {};

// State
let socket = null;
let clientId = `web-client-${Date.now()}`;
let audioChunks = [];
let isGenerating = false;
let chunksReceived = 0;
let audioContext = null;
let audioQueue = [];
let isPlaying = false;
let nextPlayTime = 0;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadVoices();
    connectToServer();
    initAudioContext();
    // Load the sample text for the initially selected voice after loading voices
    loadSampleText(); 
});

async function loadVoices() {
    try {
        const response = await fetch('voices.json');
        VOICE_DATA = await response.json();

        // Build descriptions map
        VOICE_DATA.voices.forEach(voice => {
            VOICE_DESCRIPTIONS[voice.id] = voice.description;
        });

        // Populate voice dropdown
        const voiceSelect = document.getElementById('voice');
        voiceSelect.innerHTML = '';

        VOICE_DATA.voices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice.id;
            option.textContent = `${voice.emoji} ${voice.name}`;
            voiceSelect.appendChild(option);
        });

        // Set default voice and update description
        voiceSelect.value = VOICE_DATA.defaultVoice;
        updateDescription();

        log('✅ Voice presets loaded');
    } catch (error) {
        console.error('Failed to load voices.json:', error);
        log('⚠️  Failed to load voice presets, using defaults');

        // Fallback to hardcoded
        VOICE_DESCRIPTIONS = {
            'af_heart': 'Female in their 30s with warm, friendly and conversational voice.'
        };
    }
}

function initAudioContext() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
    });
    log('🔊 Web Audio API initialized');
}

function updateDescription() {
    const voice = document.getElementById('voice').value;
    const descriptionInput = document.getElementById('description');
    descriptionInput.value = VOICE_DESCRIPTIONS[voice] || '';
    
    // Also update the sample text when the voice changes
    loadSampleText(); 
}

function loadSampleText() {
    if (!VOICE_DATA) return;

    const voice = document.getElementById('voice').value;
    const voiceData = VOICE_DATA.voices.find(v => v.id === voice);
    const textEl = document.getElementById('text');

    if (voiceData && voiceData.sampleText) {
        textEl.value = voiceData.sampleText;
        log(`📝 Loaded sample text for ${voiceData.name}`);
    } else {
        // Clear or set default text if no sample is available
        textEl.value = `Hello! This is a test of the Maya-1-Voice text-to-speech system. How do you like my voice?`;
        log(`📝 No specific sample text found, using default.`);
    }
}

function insertEmotion(emotion) {
    const textarea = document.getElementById('text');
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;
    const before = text.substring(0, start);
    const after = text.substring(end);

    // Insert emotion tag at cursor position
    const emotionTag = ` <${emotion}> `;
    textarea.value = before + emotionTag + after;

    // Place cursor after inserted tag
    const newPos = start + emotionTag.length;
    textarea.setSelectionRange(newPos, newPos);
    textarea.focus();
}

function connectToServer() {
    log('Connecting to server...');

    socket = io(SERVER_URL, {
        transports: ['websocket', 'polling']
    });

    socket.on('connect', () => {
        log('✅ Connected to server');
        updateStatus('connected', 'Connected', '🟢', 'ONLINE');
        
        // Register audio connection
        socket.emit('register_audio_connection', {
            client_id: clientId
        });
    });

    socket.on('disconnect', () => {
        log('❌ Disconnected from server');
        updateStatus('disconnected', 'Disconnected', '🔴', 'OFFLINE');
    });

    socket.on('audio_connection_registered', (data) => {
        log('🎧 Audio connection registered');
    });

    socket.on('tts_start', (data) => {
        log(`🎤 Generation started: ${data.text}`);
        // updateStatus('generating', 'Generating and playing...', '⏳', 'GENERATING');
        resetAudioPlayer();
        chunksReceived = 0;
        audioChunks = [];
        audioQueue = [];
        isPlaying = false;
        nextPlayTime = audioContext.currentTime;
    });

    socket.on('tts_audio', (data) => {
        const chunk = data.chunk || chunksReceived + 1;
        const audioBinary = data.audio;

        if (audioBinary) {
            // Socket.IO sends binary data as ArrayBuffer or Blob
            let audioBytes;
            
            if (audioBinary instanceof ArrayBuffer) {
                audioBytes = audioBinary;
            } else if (audioBinary instanceof Blob) {
                // Convert Blob to ArrayBuffer
                audioBinary.arrayBuffer().then(buffer => {
                    audioChunks.push(buffer);
                    chunksReceived++;
                    playAudioChunk(buffer);
                    if (chunk === 1) {
                        log(`⚡ First chunk received - playing now!`);
                    }
                    updateStats();
                });
                return; // Exit early for async Blob handling
            } else if (audioBinary.buffer) {
                // TypedArray (like Uint8Array)
                audioBytes = audioBinary.buffer.slice(
                    audioBinary.byteOffset, 
                    audioBinary.byteOffset + audioBinary.byteLength
                );
            } else {
                console.error('Unknown audio data type:', typeof audioBinary);
                return;
            }
            
            audioChunks.push(audioBytes);
            chunksReceived++;
            playAudioChunk(audioBytes);

            if (chunk === 1) {
                log(`⚡ First chunk received - playing now!`);
            }

            updateStats();
        }
    });

    socket.on('tts_complete', (data) => {
        log(`✅ Generation complete: ${data.chunks} chunks in ${data.duration.toFixed(2)}s`);
        updateStatus('connected', 'Connected', '🟢', 'ONLINE');
        isGenerating = false;
        enableButtons(true);
        log('🔊 Audio streaming finished');
    });

    socket.on('tts_error', (data) => {
        log(`❌ Error: ${data.error}`);
        updateStatus('connected', 'Connected', '🟢', 'ONLINE');
        isGenerating = false;
        enableButtons(true);
    });

    socket.on('tts_stop_immediate', () => {
        log('🛑 Generation stopped');
        updateStatus('connected', 'Connected', '🟢', 'ONLINE');
        isGenerating = false;
        enableButtons(true);
    });
}

function generateSpeech() {
    if (isGenerating) return;

    const voice = document.getElementById('voice').value;
    const text = document.getElementById('text').value.trim();
    const description = document.getElementById('description').value.trim();

    // --- Start of client-side validation ---
    if (!text) {
        log('⚠️ Please enter some text to speak.');
        return; // Exit function if validation fails
    }

    if (!description) {
        log('⚠️ Please enter a voice description.');
        return; // Exit function if validation fails
    }

    if (!socket || !socket.connected) {
        log('⚠️ Not connected to server.');
        return; // Exit function if validation fails
    }
    // --- End of client-side validation ---
    
    // Only disable buttons and set state AFTER validation passes
    isGenerating = true;
    enableButtons(false); 

    log(`🎵 Generating: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
    log(`📝 Description: "${description.substring(0, 50)}${description.length > 50 ? '...' : ''}"`);

    socket.emit('generate_speech', {
        client_id: clientId,
        text: text,
        voice: voice,
        description: description,  // Send custom description
        mode: 'tts'
    });
}

function stopGeneration() {
    if (!isGenerating) return;

    log('🛑 Stopping generation...');
    socket.emit('stop_current_generation', {
        client_id: clientId
    });
}

function playAudioChunk(audioBytes) {
    // Convert PCM int16 bytes to Float32Array
    const int16Array = new Int16Array(audioBytes);
    const float32Array = new Float32Array(int16Array.length);

    // Convert int16 to float32 (-1.0 to 1.0)
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }

    // Create audio buffer
    const audioBuffer = audioContext.createBuffer(1, float32Array.length, SAMPLE_RATE);
    audioBuffer.getChannelData(0).set(float32Array);

    // Create and schedule buffer source
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    // Schedule playback
    const playTime = Math.max(nextPlayTime, audioContext.currentTime);
    source.start(playTime);

    // Update next play time
    nextPlayTime = playTime + audioBuffer.duration;

    if (!isPlaying) {
        isPlaying = true;
        log('🔊 Started real-time playback');
    }
}

// Updated function signature to accept the text argument
function updateStatus(iconClassName, tooltipText, emoji, statusText) {
    const statusIconEl = document.getElementById('connectionStatusIcon');
    const statusTextEl = document.getElementById('connectionStatusText');

    if (statusIconEl) {
        // Update icon
        statusIconEl.className = iconClassName; 
        statusIconEl.title = tooltipText;
        statusIconEl.textContent = emoji;
    }

    if (statusTextEl) {
        // Update text content and class for color styling
        statusTextEl.textContent = statusText;
        statusTextEl.className = `${iconClassName}-text`; // Uses e.g., 'connected-text' for styling
    }
}

function enableButtons(enabled) {
    document.getElementById('generateBtn').disabled = !enabled;
    // document.getElementById('stopBtn').disabled = enabled;
}

function resetAudioPlayer() {
    chunksReceived = 0;
    audioQueue = [];
    isPlaying = false;
    updateStats();
}

function updateStats() {
    const totalBytes = audioChunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
    const duration = totalBytes / 2 / SAMPLE_RATE;

    document.getElementById('chunksReceived').textContent = chunksReceived;
    document.getElementById('audioSize').textContent = `${(totalBytes / 1024).toFixed(1)} KB`;
    document.getElementById('duration').textContent = `${duration.toFixed(1)}s`;
}

function log(message) {
    const logEl = document.getElementById('log');
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="time">[${time}]</span>${message}`;
    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
}