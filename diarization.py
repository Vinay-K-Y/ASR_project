from pyannote.audio import Pipeline
from huggingface_hub import login
import torchaudio
import torch
import time

def convert_audio_tensor(input_path):
    waveform, sr = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
    return waveform, sr

def run_diarization(audio_path, hf_token):
    login(hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True).to(device)

    waveform, sr = convert_audio_tensor(audio_path)
    print(f"Waveform shape: {waveform.shape}, Sample rate: {sr}")
    print(f"Audio duration (seconds): {waveform.shape[1] / sr:.2f}")
    print("Running speaker diarization...")

    start = time.time()
    diarization = pipeline({"waveform": waveform, "sample_rate": sr})
    print("Done in", round(time.time() - start, 2), "seconds")

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append((turn.start, turn.end, speaker))
    return results

