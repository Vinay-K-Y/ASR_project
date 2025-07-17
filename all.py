import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
from datetime import datetime

from transcript import transcribe_audio
from speaker_identification import load_known_speakers, identify_speaker
from diarization import run_diarization

# Config
KNOWN_SPEAKER_FOLDER = "known_speakers"
HF_TOKEN = "YOUR_TOKEN_KEY_HERE"
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "output_summary.txt"

# Initialize recording
recorded_frames = []
print("🎤 Recording... Press Ctrl+C to stop.\n")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32') as stream:
        while True:
            frame, _ = stream.read(1024)
            recorded_frames.append(frame)

except KeyboardInterrupt:
    print("\n Stopping recording...")

    # Save to temp WAV file
    audio_data = np.concatenate(recorded_frames, axis=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = f"mic_recording_{timestamp}.wav"
    sf.write(wav_path, audio_data, SAMPLE_RATE)
    print(f" Audio saved to {wav_path}\n")
    wav = "test_2.wav"
    # 1. Transcription
    print(" TRANSCRIPTION")
    transcript = transcribe_audio(wav)
    print(transcript)

    # 2. Speaker Identification
    print("\n SPEAKER IDENTIFICATION")
    known_embeddings, encoder = load_known_speakers(KNOWN_SPEAKER_FOLDER)
    speaker, similarity = identify_speaker(wav_path, known_embeddings, encoder)
    print(f"Speaker: {speaker} (Similarity: {similarity:.2f})")

    # 3. Diarization
    print("\n DIARIZATION")
    segments = run_diarization(wav_path, HF_TOKEN)
    for start, end, label in segments:
        print(f"{start:.1f}s - {end:.1f}s : {label}")

    # Save summary
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(" TRANSCRIPTION\n")
        f.write(transcript + "\n\n")
        f.write(" SPEAKER IDENTIFICATION\n")
        f.write(f"{speaker} (Similarity: {similarity:.2f})\n\n")
        f.write(" DIARIZATION\n")
        for start, end, label in segments:
            f.write(f"{start:.1f}s - {end:.1f}s : {label}\n")

    print(f"\n Results saved to {OUTPUT_FILE}")


