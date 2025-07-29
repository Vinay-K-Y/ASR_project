import torch
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
import os
import glob
from collections import defaultdict
from resemblyzer import VoiceEncoder, preprocess_wav
from vad import SileroVAD

# --- Configuration ---
# Models
DRAFT_MODEL_NAME = "tiny.en"
FINAL_MODEL_NAME = "small.en"
TRANSCRIPT_FILE = "transcript.txt" # File to save the final transcript

# Audio
SAMPLE_RATE = 16000
MIC_DEVICE_INDEX = None # Use None for default mic

# VAD & Logic
VAD_THRESHOLD = 0.35
VAD_CONFIRMATION_CHUNKS = 4
SILENCE_CONFIRMATION_CHUNKS = 15
CHUNK_SAMPLES = 512

# Speaker Diarization
EMBEDDING_THRESHOLD = 0.80

# --- Global State & Queues ---
audio_queue = queue.Queue()
draft_queue = queue.Queue()
final_queue = queue.Queue()
stop_event = threading.Event()
print_lock = threading.Lock()


# --- Speaker Diarization Functions ---
def normalize(vec):
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec

def identify_session_speaker(embed, speaker_embeddings):
    """Identify a speaker by comparing the embedding to speakers in the current session."""
    embed = normalize(embed)
    best_match = None
    highest_similarity = 0
    for name, embed_list in speaker_embeddings.items():
        smoothed = normalize(np.mean(embed_list, axis=0))
        similarity = np.dot(embed, smoothed)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name
            
    if highest_similarity > EMBEDDING_THRESHOLD:
        return best_match
    return None

# ---Thread 1: Audio Capture ---
def audio_capture_thread():
    print(" Audio capture thread started.")
    try:
        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                              blocksize=CHUNK_SAMPLES, device=MIC_DEVICE_INDEX, callback=callback):
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"Error in audio capture: {e}")
    finally:
        print(" Audio capture thread stopped.")

# -- Thread 2: VAD and Segmenter ---
def vad_and_segmenter_thread(vad_model):
    print(" VAD & Segmenter thread started.")
    is_speaking = False
    active_phrase_buffer = []
    speech_chunk_count = 0
    silence_chunk_count = 0
    draft_buffer = []

    while not stop_event.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=1)
            audio_chunk_squeezed = np.squeeze(audio_chunk)
            audio_tensor = torch.from_numpy(audio_chunk_squeezed)

            if vad_model.is_speech(audio_tensor, threshold=VAD_THRESHOLD):
                speech_chunk_count += 1
                silence_chunk_count = 0
                if not is_speaking and speech_chunk_count > VAD_CONFIRMATION_CHUNKS:
                    is_speaking = True
            else:
                silence_chunk_count += 1
                speech_chunk_count = 0
                if is_speaking and silence_chunk_count > SILENCE_CONFIRMATION_CHUNKS:
                    is_speaking = False
                    if active_phrase_buffer:
                        complete_phrase = np.concatenate(active_phrase_buffer)
                        final_queue.put(complete_phrase)
                        active_phrase_buffer = []

            if is_speaking:
                active_phrase_buffer.append(audio_chunk_squeezed)

            draft_buffer.append(audio_chunk_squeezed)
            if len(draft_buffer) * CHUNK_SAMPLES >= SAMPLE_RATE:
                draft_audio = np.concatenate(draft_buffer)
                draft_queue.put(draft_audio)
                draft_buffer = []

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in VAD thread: {e}")
            break
    print(" VAD & Segmenter thread stopped.")

# --- ⚡ Thread 3: Draft Transcription ---
def draft_transcription_thread(model, device, lock):
    print("⚡ Draft transcription thread started.")
    prompt_context = ""
    while not stop_event.is_set():
        try:
            chunk = draft_queue.get(timeout=1)
            audio_tensor = torch.from_numpy(chunk).float()
            
            padded_audio = whisper.pad_or_trim(audio_tensor, length=whisper.audio.N_SAMPLES)
            mel = whisper.log_mel_spectrogram(padded_audio).to(device)

            options = whisper.DecodingOptions(
                language="en", without_timestamps=True, fp16=(device=="cuda"), prompt=prompt_context
            )
            result = whisper.decode(model, mel, options)

            with lock:
                print(f"\r Draft: {result.text.strip()}", end="", flush=True)
            
            prompt_context = result.text

        except queue.Empty:
            continue
        except Exception as e:
            print(f" Error in draft thread: {e}")
            break
    print(" Draft transcription thread stopped.")

# --Thread 4: Final Transcription & Diarization ---
def final_transcription_thread(model, device, encoder, lock):
    print("Final transcription & diarization thread started.")
    session_speaker_embeddings = defaultdict(list)
    speaker_counter = 1
    
    while not stop_event.is_set():
        try:
            phrase = final_queue.get(timeout=1)
            
            wav = preprocess_wav(phrase, source_sr=SAMPLE_RATE)
            embed = encoder.embed_utterance(wav)
            
            speaker_name = identify_session_speaker(embed, session_speaker_embeddings)
            
            if speaker_name is None:
                speaker_name = f"Speaker {speaker_counter}"
                speaker_counter += 1
            session_speaker_embeddings[speaker_name].append(normalize(embed))
            
            audio_tensor = torch.from_numpy(phrase).float()
            padded_audio = whisper.pad_or_trim(audio_tensor, length=whisper.audio.N_SAMPLES)
            mel = whisper.log_mel_spectrogram(padded_audio).to(device)

            options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=(device=="cuda"))
            result = whisper.decode(model, mel, options)
            transcript_text = result.text.strip()

            if transcript_text:
                with lock:
                    # Clear the draft line before printing the final transcript
                    print(f"\r{' ' * 100}\r", end="") 
                    
                    formatted_transcript = f"🗣️  {speaker_name}: {transcript_text}"
                    print(formatted_transcript)
                    
                    # Save the final transcript to a file
                    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                        f.write(formatted_transcript + "\n")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in final thread: {e}")
            break
    print("Final transcription thread stopped.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting real-time transcription...")
    torch.set_num_threads(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, using CPU. Performance will be severely limited.")
    else:
        print("CUDA is available. Using GPU.")

    print(f"Loading '{DRAFT_MODEL_NAME}' model for drafts...")
    draft_model = whisper.load_model(DRAFT_MODEL_NAME, device=device)
    
    print(f"Loading '{FINAL_MODEL_NAME}' model for final transcription...")
    final_model = whisper.load_model(FINAL_MODEL_NAME, device=device)
    
    print("Loading Resemblyzer Voice Encoder...")
    encoder = VoiceEncoder(device=device)
    
    print("Loading Silero VAD model...")
    vad_model = SileroVAD()
    
    print(f"All models loaded.")
    
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(f"--- Transcript session started at {time.ctime()} ---\n")
    print(f"📝 Final transcript will be saved to '{TRANSCRIPT_FILE}'")

    
    threads = [
        threading.Thread(target=audio_capture_thread, daemon=True),
        threading.Thread(target=vad_and_segmenter_thread, args=(vad_model,), daemon=True),
        threading.Thread(target=draft_transcription_thread, args=(draft_model, device, print_lock), daemon=True),
        threading.Thread(target=final_transcription_thread, args=(final_model, device, encoder, print_lock), daemon=True)
    ]

    for t in threads:
        t.start()

    print("\n Speak into your microphone. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        stop_event.set()

    for t in threads:
        t.join()

    print(f"\nFinal transcript saved in '{TRANSCRIPT_FILE}'.")
