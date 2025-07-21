import sounddevice as sd
import numpy as np
import torch
import whisper
import queue
import time
from resemblyzer import VoiceEncoder, preprocess_wav
import os
import glob
from vad import SileroVAD  # your local wrapper file
vad = SileroVAD()

# Load models
print("Loading Whisper and Resemblyzer...")
whisper_model = whisper.load_model("base")
encoder = VoiceEncoder()
print("✅ Models loaded.")

# Settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
EMBEDDING_THRESHOLD = 0.75  # cosine similarity

# Prepare buffer and queue
q_audio = queue.Queue()

# Load known speaker embeddings
def load_known_speakers(known_dir="known_speakers"):
    speaker_embeddings = {}
    for person_dir in os.listdir(known_dir):
        person_path = os.path.join(known_dir, person_dir)
        if os.path.isdir(person_path):
            embeddings = []
            for file in glob.glob(os.path.join(person_path, "*.wav")):
                wav = preprocess_wav(file)
                embed = encoder.embed_utterance(wav)
                embeddings.append(embed)
            avg_embed = np.mean(embeddings, axis=0)
            speaker_embeddings[person_dir] = avg_embed
    return speaker_embeddings

known_speakers = load_known_speakers()
print(f"🧠 Loaded {len(known_speakers)} known speakers.")

# Identify known speaker
def identify_known_speaker(embed):
    best_match = None
    highest_similarity = 0
    for name, known_embed in known_speakers.items():
        similarity = np.dot(embed, known_embed) / (np.linalg.norm(embed) * np.linalg.norm(known_embed))
        if similarity > highest_similarity:
            best_match = name
            highest_similarity = similarity
    if highest_similarity > EMBEDDING_THRESHOLD:
        return best_match
    return None

# Audio callback
def callback(indata, frames, time_info, status):
    if status:
        print(status)
    q_audio.put(indata.copy())

# Check if speech is present using VAD
def is_speech(chunk):
    audio = np.squeeze(chunk).astype(np.float32)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    return vad.is_speech(audio_tensor)


# Start stream
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
stream.start()

print("🎙️ Speak into the microphone... (Press Ctrl+C to stop)")

# State variables
current_speaker = None
unknown_speaker_embeddings = {}
unknown_counter = 1

try:
    buffer = np.zeros((0, 1), dtype=np.float32)

    while True:
        while not q_audio.empty():
            buffer = np.concatenate((buffer, q_audio.get()))

        if len(buffer) < CHUNK_SIZE:
            continue

        chunk = buffer[:CHUNK_SIZE]
        buffer = buffer[CHUNK_SIZE:]

        if not is_speech(chunk):
            continue  # skip non-speech

        samples = np.squeeze(chunk)
        wav = preprocess_wav(samples, source_sr=SAMPLE_RATE)
        embed = encoder.embed_utterance(wav)

        # Try to identify known speaker
        speaker_name = identify_known_speaker(embed)

        # If not known, try matching with previous unknowns
        if speaker_name is None:
            matched_label = None
            for label, unk_embed in unknown_speaker_embeddings.items():
                similarity = np.dot(embed, unk_embed) / (np.linalg.norm(embed) * np.linalg.norm(unk_embed))
                if similarity > EMBEDDING_THRESHOLD:
                    matched_label = label
                    break

            if matched_label:
                speaker_name = matched_label
            else:
                speaker_name = f"Speaker {unknown_counter}"
                unknown_speaker_embeddings[speaker_name] = embed
                unknown_counter += 1

        # Transcribe
        audio_for_whisper = whisper.pad_or_trim(torch.from_numpy(samples).float())
        mel = whisper.log_mel_spectrogram(audio_for_whisper).to(whisper_model.device)
        options = whisper.DecodingOptions(language="en", fp16=False)
        result = whisper.decode(whisper_model, mel, options)
        text = result.text.strip()

        if not text:
            continue

        if speaker_name != current_speaker:
            print(f"\n🗣️ {speaker_name}: ", end="", flush=True)
            current_speaker = speaker_name

        print(text + " ", end="", flush=True)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    stream.stop()
    stream.close()





