import sounddevice as sd
import numpy as np
import queue
import time
import torch
from vad import SileroVAD  # Optional
from riva.client import ASRService
from riva.client.audio_io import AudioChunk

# Riva client
riva_asr = ASRService(
    url="localhost:50051",  # Change if using remote Riva
    insecure=True
)

SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
vad = SileroVAD()
q_audio = queue.Queue()

# Riva ASR with speaker diarization enabled
config = riva_asr.get_config(
    language_code="en-US",
    sample_rate_hz=16000,
    enable_automatic_punctuation=True,
    enable_word_time_offsets=False,
    enable_speaker_diarization=True  # ✅ Enable speaker labeling
)

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    q_audio.put(indata.copy())

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
stream.start()

print("🎙️ Real-time Riva ASR + Diarization (Ctrl+C to stop)")
buffer = np.zeros((0,), dtype=np.float32)

try:
    while True:
        while not q_audio.empty():
            data = q_audio.get()
            buffer = np.concatenate((buffer, data.squeeze()))

        if len(buffer) < CHUNK_SIZE:
            continue

        chunk = buffer[:CHUNK_SIZE]
        buffer = buffer[CHUNK_SIZE:]  # Remove processed part

        if not vad.is_speech(torch.from_numpy(chunk).unsqueeze(0)):
            continue  # Skip silence

        # Convert to int16 bytes for Riva
        audio_bytes = (chunk * 32767).astype(np.int16).tobytes()

        # Get Riva ASR results (with speaker labels)
        responses = riva_asr.streaming_response_generator(
            audio_chunks=[AudioChunk(audio_bytes)],
            config=config
        )

        for response in responses:
            for result in response.results:
                if result.alternatives:
                    transcript = result.alternatives[0].transcript.strip()
                    speaker = result.speaker_id or "Unknown"
                    if transcript:
                        print(f"\n🗣️ Speaker {speaker}: {transcript}")

except KeyboardInterrupt:
    print("\n🛑 Exiting.")
    stream.stop()
    stream.close()
