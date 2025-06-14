import streamlit as st
import os
import torch
from pydub import AudioSegment
import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier
import tempfile
import numpy as np
import re
import subprocess


# Initialize models (cached for performance)
@st.cache_resource
def load_accent_model():
    return EncoderClassifier.from_hparams(
        source='speechbrain/acc-identification-commonaccent_ecapa',
        savedir='pretrained_models'
    )

def download_video(url, output_file='input_video.mp4'):
    """Download video from YouTube or a direct link with yt-dlp."""
    try:
        cmd = ['yt-dlp', '-v', '-f', 'best', '-o', output_file, url]
        subprocess.run(cmd, check=True)
        return output_file
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None



def extract_audio(video_file):
    """Extract audio from video without duration limit."""
    try:
        audio_file = video_file.rsplit('.', 1)[0] + ".wav"

        # Extract audio with ffmpeg
        cmd = ['ffmpeg', '-i', video_file, '-ac', '1', '-ar', '16000', '-y', audio_file]
        subprocess.run(cmd, check=True)

        return audio_file

    except Exception as e:
        st.error(f"Audio extraction error: {str(e)}")
        return None


def transcribe_audio(wav_file):
    """Transcribe audio to text with Speech Recognition."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except Exception as e:
        st.error(f"Transcribe error: {str(e)}")
        return ''


def classify_accent(wav_file, model):
    """Classify the accent of the speaker in the audio file."""
    try:
        out_prob, score, index, text_lab = model.classify_file(wav_file)
        accent_code = text_lab[0]
        confidence = torch.exp(score).item() * 100

        # Map accent codes to human-readable names
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


# Streamlit UI
st.title("🎤 English Accent Detection")
st.write("""
Analyze the speaker's English accent from a video (direct link or YouTube).
**Note:** Large files may take a while to process.
""")

input_url = st.text_input("Enter video URL (direct or YouTube)", 
                           placeholder='https://www.youtube.com/watch?v=sGvHsGvcrGQ')

if st.button("Analyze Accent"):

    if not input_url:
        st.error("Please provide a video URL.")
    else:
        with st.spinner("Downloading video..."):
            video_file = download_video(input_url)
            if not video_file:
                st.stop()

        with st.spinner("Extracting audio from video..."):
            wav_file = extract_audio(video_file)
            if not wav_file:
                st.stop()

        st.audio(wav_file)

        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(wav_file)
            if not transcription:
                st.error("No English speech detected in the audio.")
                st.stop()

        with st.spinner("Classifying accent..."):
            model = load_accent_model()
            accent, confidence = classify_accent(wav_file, model)

        st.success("Analysis complete.")
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        col1.metric("Detected Accent", accent)
        col2.metric("Confidence Score", f"{confidence:.1f}%")

        st.progress(int(confidence))

        st.subheader("Summary")
        st.write(f"""- 🗣 **Transcription**: {transcription}
- 🌟 **Predicted Accent**: {accent}
- ✅ **Confidence**: {confidence:.1f}%""")

st.caption("Built with ❤️ using Streamlit, SpeechBrain, and yt-dlp")
