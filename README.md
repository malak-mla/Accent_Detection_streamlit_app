# 🎤 English Accent Detection

Analyze the speaker's English accent from a video (direct link or YouTube).  
This tool is designed to aid in evaluating a candidate’s English pronunciation for hiring purposes.

---

## Features:

✅ Accepts **direct video URLs or YouTube URLs** (standard or shorts).  
✅ Extracts audio and parses the speaker’s pronunciation.  
✅ Detects the English accent (British, American, Australian, etc.).  
✅ Shows confidence score alongside the detection.  
✅ Displays a brief transcript of the audio.

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

git clone <your-repo-url>
cd <your-repo-name>

pip install -r requirements.txt

##  Installing FFMPEG (required by pydub and yt-dlp):

sudo apt install ffmpeg

## Running:
streamlit run app.py
## Developer Notes:
- This tool utilizes yt-dlp for video downloading.

- Audio is processed with pydub.

- Seech recognition is powered by Speech Recognition’s recognize_google.

- Accent detection is powered by SpeechBrain’s ECAPA classifier.


