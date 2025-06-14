import streamlit as st
import requests
import os
import torch
from pydub import AudioSegment
import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier
import tempfile
import numpy as np
import re
import io
import subprocess

# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
@st.cache_resource
def load_accent_model():
    """Load SpeechBrain ECAPA classifier for English accents."""
    return EncoderClassifier.from_hparams(
        source='speechbrain/acc-identification-commonaccent_ecapa',
        savedir='pretrained_models'
    )

# ------------------------------------------------------------------------------
# File Operations
# ------------------------------------------------------------------------------
def transform_url(url):
    """Transform YouTube Short URLs to standard URLs if applicable."""
    if "youtu.be" in url or "/shorts/" in url:
        video_id = url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def download_video(url, output_file='input_video.mp4'):
    """Download video from YouTube or a direct link with yt-dlp."""
    try:
        url = transform_url(url)
        cmd = ['yt-dlp', '-v', '-f', 'best', '-o', output_file, url]
        subprocess.run(cmd, check=True)
        return output_file
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None


def extract_audio(video_file):
    """Extract audio from video and convert to a standardized format."""
    try:
        audio_file = video_file.split('.')[0] + '.wav'
        audio = AudioSegment.from_file(video_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio = audio.normalize()
        audio.export(audio_file, format='wav')
        return audio_file, len(audio) / 1000
    except Exception as e:
        st.error(f"Audio extraction error: {str(e)}")
        return None, 0


def transcribe_audio(wav_file):
    """Transcribe the audio to text with Speech Recognition."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            st.error(f"Speech recognition error: {str(e)}")
            return ""

def classify_accent(wav_file, model):
    """Classify the speaker's English accent with ECAPA classifier."""
    try:
        out_prob, score, index, text_lab = model.classify_file(wav_file)
        accent_code = text_lab[0]
        confidence = torch.exp(score).item() * 100

        # Mapping to human-readable names
        accent_map = {
            'au': 'Australian',
            'ca': 'Canadian',
            'uk': 'British',
            'us': 'American',
            'ind': 'Indian',
            'afr': 'African',
            'sp': 'Spanish'
        }

        return accent_map.get(accent_code, accent_code), min(confidence, 100)
    except Exception as e:
        st.error(f"Accent classification error: {str(e)}")
        return "Error", 0.0


# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------
st.title("🎤 English Accent Detection")
st.write("""
Analyze the speaker's English accent from a video (direct link or YouTube).
""")


# Input for video URL
video_url = st.text_input("Enter video URL (direct or YouTube)", 
                           placeholder='https://www.youtube.com/watch?v=abcd1234')

if st.button("Analyze Accent"):
    if not video_url:
        st.error("Please enter a video URL.")
    else:
        with st.spinner("Downloading video..."):
            video_file = download_video(video_url)
            if video_file is None:
                st.stop()

        with st.spinner("Extracting audio..."):
            wav_file, duration = extract_audio(video_file)
            if wav_file is None:
                st.stop()

            st.audio(wav_file)
            st.caption(f"Audio duration: {duration:.1f} seconds")

        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(wav_file)
            if not transcription:
                st.error("No English speech detected.")
                st.stop()

        with st.spinner("Classifying accent..."):
            model = load_accent_model()
            accent, confidence = classify_accent(wav_file, model)

        st.success("Analysis complete.")
        st.subheader("Analysis Result")
        st.metric("Detectable Accent", accent)
        st.metric("Confidence Score", f'{confidence:.1f}%')
        st.text_area("Transcription", transcription)

        st.write(f"""  
- 📝 Transcription:  
{transcription}

- 🔹 Accent Detected: {accent}
- 🔹 Confidence Score: {confidence:.1f}%
- 🔹 Audio Duration: {duration:.1f} seconds
""")


st.caption("Built with ❤️ using Streamlit, SpeechBrain, Speech Recognition, pydub, and yt-dlp")
