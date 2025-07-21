# vad.py
import torch

class SileroVAD:
    def __init__(self):
        print("⏳ Loading Silero VAD...")

        # Load the model and unpack the utils tuple
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )

        self.model = model
        self.get_speech_ts, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
        self.sample_rate = 16000

        print("✅ Silero VAD Loaded.")

    def is_speech(self, audio_tensor):
        """
        Returns True if speech is detected in the audio tensor.
        """
        speech_timestamps = self.get_speech_ts(audio_tensor, self.model, sampling_rate=self.sample_rate)
        return len(speech_timestamps) > 0



