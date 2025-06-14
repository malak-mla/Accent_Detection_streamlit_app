# 🌍 English Accent Detection App

This application analyzes video content to detect English accents and evaluate spoken English proficiency. Built for REM Waste's hiring evaluation process, the tool automates accent classification by:

- Accepting public video URLs (MP4 and Loom video URLs)

- Extracting audio content from video

- Analyzing speaker accent

- Providing detailed results with confidence scores

The solution uses Python with Streamlit for the frontend and leverages cutting-edge speech recognition models from SpeechBrain for accurate accent classification.
## Features
🎥 Public video URL processing - Accepts links from Loom, Dropbox, or any direct video source

🔊 Audio extraction and preview - Converts video to playable audio

🌍 Accent classification - Identifies 7 major English accents

📊 Confidence scoring - Provides 0-100% confidence metric

📝 Transcription verification - Ensures English content before classificatio

## Technical Highlights


⚡ Model caching - Fast reloads after initial setup

🧹 Automatic cleanup - Temporary file management

🛡️ Comprehensive error handling - User-friendly error messages

📏 Resource limits - 20MB file size, 60s audio processing

📱 Responsive UI - Works on desktop and mobile
 
 ## How It Works
### 1- Input Handling :
- User provides public video URL

- System validates URL and file size (<20MB)

### 2-Audio Extraction:

- Video downloaded to temp storage

- Audio converted to 16kHz mono WAV format

- Truncated to first 60 seconds

### 3-Speech Recognition:

- Google Web Speech API transcribes audio

- Verifies English content exists

### 4-Accent Classification:
- SpeechBrain's ECAPA-TDNN model analyzes audio

- Identifies accent from 7 supported types

- Generates confidence score

### 4-Result Presentation:
- Audio preview player

- Accent classification card

- Confidence score with progress bar

- Detailed analysis summary

## Installation
- Python 3.8+

- FFmpeg (for audio processing)


## Setup
```bash
python -m venv detectenv
source detectenv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

