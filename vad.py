import torch

class SileroVAD:
    def __init__(self):
        print(" Loading Silero VAD...")

        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )

        self.model = model
        self.get_speech_ts, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
        self.sample_rate = 16000

        print(" Silero VAD Loaded.")

    def is_speech(self, audio_tensor, threshold=0.5):
        """
        This is a more direct and responsive method for real-time VAD.
        It returns the speech probability of the LAST frame in the audio tensor.
        """
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.from_numpy(audio_tensor)

        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        return speech_prob > threshold


