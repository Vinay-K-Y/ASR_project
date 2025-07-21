from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def load_known_speakers(folder_path):
    encoder = VoiceEncoder()
    embeddings = {}

    for speaker_dir in Path(folder_path).iterdir():
        if speaker_dir.is_dir():
            speaker_name = speaker_dir.name
            all_embeds = []
            for audio_file in speaker_dir.glob("*.wav"):
                wav = preprocess_wav(str(audio_file))
                emb = encoder.embed_utterance(wav)
                all_embeds.append(emb)
            if all_embeds:
                embeddings[speaker_name] = np.mean(all_embeds, axis=0)
    return embeddings

def identify_speaker(test_audio_path, known_embeddings, encoder, threshold=0.75):
    test_wav = preprocess_wav(test_audio_path)
    test_embed = encoder.embed_utterance(test_wav)

    max_sim = -1
    identity = "unknown"
    for name, known_emb in known_embeddings.items():
        sim = cosine_similarity(test_embed, known_emb)
        if sim > max_sim and sim > threshold:
            max_sim = sim
            identity = name
    return identity, max_sim

