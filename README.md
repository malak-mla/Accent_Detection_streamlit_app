# 🎤 English Accent Detection

Analyze the English accent from a video or audio file — a tool designed for recruiters or organizations that want to aid their decisions with automated, data-informed assessments

---
## Overview:
This Streamlit application performs the following:

- Downloads a video or audio file from a direct URL.

- Extracts its audio track.

- Transcribes the speech to text.

- Detects the English accent of the speaker (British, American, Canadian, Australian, etc.).

- Displays a confidence score alongside the transcription.

## Features:

+ ✅ Transcribes the audio to text (using Speech Recognition’s Google API)
- ✅ Detects the English accent with SpeechBrain’s ECAPA classifier
- ✅ Displays confidence score and audio duration
- ✅ Streamlit UI for easy interaction — no coding needed to use the tool
- ✅ Audio preview directly within Streamlit

---

## Tech Stack:

- **Python 3.10+**
- Streamlit (for UI)
- yt-dlp (for video download)
- SpeechBrain (for accent detection)
- Speech Recognition (for transcription)
- pydub (for audio processing)

---
### Installation:

To run this application, make sure you have Python 3.8+ installed.

### 1-Clone or download this repository:

git clone <your-repository-url>

cd your-repository-name

### 2-Create a virtual environment (optional but recommended):

python -m venv .venv

source .venv/bin/activate  # Linux/Mac

.\venv\Scripts\activate   # Windows

##  Installing of Requirements:

pip install -r requirements.txt

## Running:

streamlit run app.py

Then follow the instructions in your browser:

- Enter a direct video or audio URL in the text box.

- Click Analyze Accent.

- Wait for the process to complete.

- View the transcription, confidence score, and extracted audio preview.

##  File Format Support
- ✅ Video (MP4, WEBM, MKV, etc)
- ✅ Audio (MP3, WAV, etc)

##  Model
- This application utilizes SpeechBrain’s ECAPA classifier for English accent detection.

##  Notes:

- Large files may take a while to process.

- The application is for internal and educational use.

- Transcription depends on the audio’s clarity and the speaker’s pronunciation.

## Built With :

+ Python

- Streamlit

- SpeechBrain

- Speech Recognition

- pydub