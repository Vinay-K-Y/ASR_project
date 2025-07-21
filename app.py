import streamlit as st
import os
import tempfile
import io
import sys

from all import diarization_based_transcription

# ✅ Hardcoded Hugging Face Token — Replace this with your real one
HF_TOKEN = "ur_hf_token"

st.title("🗣️ Speaker Diarization + Transcription App")

st.markdown("""
Upload a WAV audio file and get speaker diarization, speaker ID, and transcription.
""")

uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    st.audio(tmp_path, format="audio/wav")
    st.info("Processing audio — this may take a while...")

    # Redirect print to Streamlit
    class StreamlitLogger(io.StringIO):
        def __init__(self):
            super().__init__()
            self.lines = []
        def write(self, txt):
            self.lines.append(txt)
        def flush(self):
            pass
        def get_text(self):
            return "".join(self.lines)

    logger = StreamlitLogger()
    old_stdout = sys.stdout
    sys.stdout = logger

    try:
        diarization_based_transcription(tmp_path, HF_TOKEN)
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
    finally:
        sys.stdout = old_stdout
    
    st.text_area("📄 Transcription Output", value=logger.get_text(), height=400)

    # Clean up temp file
    os.remove(tmp_path)
else:
    st.info("Upload a WAV audio file to get started.")
