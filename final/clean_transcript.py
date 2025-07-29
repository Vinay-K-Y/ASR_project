import sys
import re
import subprocess
from pathlib import Path
from typing import Optional

OLLAMA_MODEL = "mistral"
DEFAULT_OUTPUT_FILE = "diarized_transcript.txt"

def load_transcript(file_path: str) -> str:
    try:
        return Path(file_path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' could not be found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        sys.exit(1)

def save_transcript(output: str, file_path: str):
    try:
        Path(file_path).write_text(output, encoding="utf-8")
        print(f"Success! Cleaned transcript saved to '{file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")
        sys.exit(1)

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return '\n'.join(sentences)

def generate_diarization_prompt(raw_text: str) -> str:
    return f"""
You are an expert AI assistant that specializes in correcting and formatting messy conversation transcripts.

**Your Task:**
Your goal is to take a raw, unstructured block of text and format it into a clean, turn-by-turn dialogue.

**Critical Instructions:**
1.  There are only **two** speakers. You must label them "Speaker 1" and "Speaker 2".
2.  The first person to speak is always "Speaker 1".
3.  A change in speaker usually happens after a question is asked or when the topic shifts. Pay close attention to questions (ending in "?") and their subsequent answers.
4.  Each distinct conversational turn should be on a new line, starting with the correct speaker label (e.g., "Speaker 1:").
5.  If the transcript has a metadata header (like a timestamp), keep it on its own line at the very top.
6.  Correct obvious grammatical mistakes and punctuation to improve readability.
7.  Return ONLY the final, formatted transcript. Do not add any extra comments, introductions, or summaries.

---
**## Example of Your Task**

**Input Example:**
--- Transcript session started at Mon Jul 28 10:30:00 2025 ---
hey how did the presentation go you looked a little nervous up there. It went okay I think. the demo part was a bit rocky but I recovered. what did you think? honestly it was great the content was strong and you handled the tough questions from marketing really well. thanks i appreciate that.

**Desired Output Example:**
--- Transcript session started at Mon Jul 28 10:30:00 2025 ---
Speaker 1: Hey, how did the presentation go? You looked a little nervous up there.
Speaker 2: It went okay, I think. The demo part was a bit rocky, but I recovered. What did you think?
Speaker 1: Honestly, it was great. The content was strong, and you handled the tough questions from marketing really well.
Speaker 2: Thanks, I appreciate that.
---

**Now, apply these rules to the following transcript.**

**Raw Transcript to Process:**
{raw_text}

**Cleaned and Diarized Transcript:**
"""

def process_with_ollama(transcript: str) -> Optional[str]:
    prompt = generate_diarization_prompt(transcript)
    print("Sending transcript to Ollama for processing. This may take a moment...")
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8"
        )
        print("Processing complete.")
        return result.stdout.strip()
    except FileNotFoundError:
        print("Error: The 'ollama' command was not found.")
        print("   Please make sure Ollama is installed and accessible in your system's PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Ollama: {e}")
        print(f"   Ollama's error output:\n{e.stderr}")
        return None

def main():
    print("--- Transcript Diarization Tool ---")
    input_file = input("Enter the path to your raw transcript file: ").strip()
    raw_transcript = load_transcript(input_file)
    header = ""
    body = raw_transcript
    lines = raw_transcript.splitlines()
    if lines and "---" in lines[0]:
        header = lines[0] + "\n"
        body = "\n".join(lines[1:])
    preprocessed_body = preprocess_text(body)
    final_input_for_model = header + preprocessed_body
    cleaned_transcript = process_with_ollama(final_input_for_model)
    if cleaned_transcript:
        save_transcript(cleaned_transcript, file_path=DEFAULT_OUTPUT_FILE)

if __name__ == "__main__":
    main()
