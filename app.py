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


# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
@st.cache_resource
def load_accent_model():
    """Load SpeechBrain's Accent Classification Model with Caching."""
    return EncoderClassifier.from_hparams(
        source='speechbrain/acc-identification-commonaccent_ecapa',
        savedir='pretrained_models'
    )


# ------------------------------------------------------------------------------
# File Operations
# ------------------------------------------------------------------------------
def download_file(url: str):
    """Download a video or audio file from a URL."""
    try:
        if not re.match(r'^https?://', url, re.I):
            raise ValueError("Invalid URL format")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save to a temporary file in memory
        file_bytes = io.BytesIO()
        for chunk in response.iter_content(8192):
            file_bytes.write(chunk)
        file_bytes.seek(0)

        # Provide a fallback for naming
        fname = url.split("/")[-1].split("?")[0] or "input_video.mp4"

        return fname, file_bytes

    except Exception as e:
        st.error(f"Error downloading the video: {str(e)}")
        return None, None


def extract_audio(file_bytes, fname):
    """Extract Audio from downloaded video or audio."""
    try:
        # Store temporarily to extract audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1]) as tmp_file:
            tmp_file.write(file_bytes.read()) 
            tmp_file_path = tmp_file.name

        # Extract and convert
        audio = AudioSegment.from_file(tmp_file_path)
        os.unlink(tmp_file_path)

        audio = audio.set_channels(1).set_frame_rate(16000)
        audio = audio.normalize()

        # Export back to memory
        wav_bytes = io.BytesIO()
        audio.export(wav_bytes, format='wav')
        wav_bytes.seek(0)

        return wav_bytes, len(audio)

    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None, 0


# ------------------------------------------------------------------------------
# Transcribe Audio
# ------------------------------------------------------------------------------
def transcribe_audio(wav_bytes):
    """Transcribe Audio to Text with Speech Recognition (Google) API."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav.write(wav_bytes.read()) 
        tmp_wav_file = tmp_wav.name

    try:
        with sr.AudioFile(tmp_wav_file) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
    except Exception as e:
        st.error(f"Transcribe Audio Error: {str(e)}")
        transcription = ""

    os.unlink(tmp_wav_file)
    return transcription


# ------------------------------------------------------------------------------
# Accent Classification
# ------------------------------------------------------------------------------
def classify_accent(wav_bytes, model):
    """Classify Accent using SpeechBrain's ECAPA Model."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(wav_bytes.read()) 
            tmp_wav_file = tmp_wav.name

        out_prob, score, index, text_lab = model.classify_file(tmp_wav_file)
        accent_code = text_lab[0]
        confidence = float(torch.exp(score).item()) * 100

        os.unlink(tmp_wav_file)

        # Mapping codes to human-readable names
        accent_map = {
            'au': 'Australian',
            'ca': 'Canadian',
            'uk': 'British',
            'us': 'American',
            'ind': 'Indian',
            'afr': 'African',
            'esp': 'Spanish'
        }

        return accent_map.get(accent_code, f"Unknown: {accent_code}"), min(confidence, 100)

    except Exception as e:
        st.error(f"Accent Classification Error: {str(e)}")
        return "Error", 0.0


# ------------------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------------------
st.title("🎤 English Accent Detection")
st.write("""
Analyze the English accent from a video or audio by providing its URL.
This tool is used internally by recruiters to aid their decisions.
""")

# Input
video_url = st.text_input("Enter video URL (public)", placeholder='https://example.com/video.mp4')

# Action
if st.button("Analyze Accent"):

    if not video_url:
        st.error("Please enter a video URL.")
    else:
        with st.spinner("Downloading video..."):
            fname, file_bytes = download_file(video_url)
            if not fname or not file_bytes:
                st.error("Unable to proceed.")
                st.stop()

        with st.spinner("Extracting audio..."):
            wav_bytes, audio_duration = extract_audio(file_bytes, fname)
            if wav_bytes is None:
                st.error("Unable to extract audio.")
                st.stop()

            duration_seconds = audio_duration / 1000
            wav_bytes.seek(0)  # Reset stream

            st.audio(wav_bytes, format='audio/wav')
            st.success(f"Audio extracted. Duration: {duration_seconds:.1f} seconds")

        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(io.BytesIO(wav_bytes.getvalue()))
            if not transcription:
                st.error("Unable to transcribe.")
                st.stop()

        with st.spinner("Classifying accent..."):
            model = load_accent_model()
            # We need a new copy for classifier
            wav_bytes_new = io.BytesIO(wav_bytes.getvalue()) 
            accent, confidence = classify_accent(wav_bytes_new, model)

        st.success("Analysis complete.")
        st.subheader("Analysis Summary")
        st.write(f"- 📝 Transcription (excerpt): {transcription[:100]}{'...' if len(transcription)>100 else ''}")
        st.write(f"- 🏹 Detected Accent: **{accent}**")
        st.write(f"- 🔮 Confidence Score: **{confidence:.1f}%**")
        st.write(f"- ⏱ Audio Duration: **{duration_seconds:.1f} seconds**")

        st.progress(min(int(confidence), 100))


st.caption("Built with ❤️ using Streamlit, SpeechBrain, and Speech Recognition.")
