from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

def transcribe_audio(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

    speech, sr = torchaudio.load(audio_path)

    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)

    speech = speech.squeeze()
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

