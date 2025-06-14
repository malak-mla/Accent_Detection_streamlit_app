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

# Initialize models (cached for performance)
@st.cache_resource
def load_accent_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/acc-identification-commonaccent_ecapa",
        savedir="pretrained_models"
    )

def download_file(url):
    """Download file from URL without size limit"""
    try:
        # Verify URL format
        if not re.match(r'^https?://', url, re.I):
            raise ValueError("Invalid URL format")
            
        # Get filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = "video.mp4"
            
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file (HTTP {response.status_code})")
        
        # Use BytesIO to avoid temporary files
        file_bytes = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            file_bytes.write(chunk)
        file_bytes.seek(0)
        
        return file_bytes, filename
        
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None, None

def extract_audio(file_bytes, filename):
    """Extract audio from video without duration limit"""
    try:
        # Create temporary file-like object
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_bytes.read())
            tmp_path = tmp_file.name
            
        # Load audio from temporary file
        audio = AudioSegment.from_file(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Convert to mono and resample
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Normalize audio volume to improve speech detection
        audio = audio.normalize()
        
        # Convert to WAV in memory
        wav_bytes = io.BytesIO()
        audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)
        
        return wav_bytes, len(audio)
        
    except Exception as e:
        st.error(f"Audio extraction error: {str(e)}")
        return None, 0

def transcribe_audio(wav_bytes):
    """Transcribe audio and return text with enhanced detection"""
    recognizer = sr.Recognizer()
    
    # Adjust energy threshold for better detection
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    
    # Create temporary WAV file for recognition
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav.write(wav_bytes.read())
        tmp_wav_path = tmp_wav.name
    
    try:
        with sr.AudioFile(tmp_wav_path) as source:
            # Listen with longer timeout
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except sr.WaitTimeoutError:
        return ""
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return ""
    finally:
        # Clean up temporary file
        os.unlink(tmp_wav_path)

def classify_accent(wav_bytes, model):
    """Classify accent and return results"""
    try:
        # Create temporary file for classification
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(wav_bytes.read())
            tmp_wav_path = tmp_wav.name
        
        # Get classification
        out_prob, score, index, text_lab = model.classify_file(tmp_wav_path)
        accent_code = text_lab[0]
        confidence = torch.exp(score).item() * 100
        
        # Clean up temporary file
        os.unlink(tmp_wav_path)
        
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
Upload a public video URL to analyze the speaker's English accent.
**Note:** Large files may take longer to process.
""")

video_url = st.text_input("Enter video URL:", placeholder="https://example.com/video.mp4")

if st.button("Analyze Accent"):
    if not video_url:
        st.warning("Please enter a valid video URL")
    else:
        with st.spinner("Downloading video..."):
            file_bytes, filename = download_file(video_url)
            if file_bytes is None:
                st.stop()
                
        with st.spinner("Extracting audio..."):
            wav_bytes, audio_duration = extract_audio(file_bytes, filename)
            if wav_bytes is None:
                st.stop()
                
            duration_seconds = audio_duration / 1000
            wav_bytes.seek(0)  # Reset pointer
            
            # Audio preview
            st.subheader("Extracted Audio")
            st.audio(wav_bytes, format='audio/wav')
            st.caption(f"Audio duration: {duration_seconds:.1f} seconds")
            
        with st.spinner("Detecting speech..."):
            # Make a copy for transcription
            transcribe_bytes = io.BytesIO(wav_bytes.getvalue())
            transcription = transcribe_audio(transcribe_bytes)
            
            if not transcription:
                st.error("No English speech detected in the audio")
                st.stop()
                
        with st.spinner("Analyzing accent..."):
            # Make a copy for classification
            classify_bytes = io.BytesIO(wav_bytes.getvalue())
            model = load_accent_model()
            accent, confidence = classify_accent(classify_bytes, model)
            
        # Display results
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        col1.metric("Detected Accent", accent)
        col2.metric("Confidence Score", f"{confidence:.1f}%")
        
        st.progress(int(confidence))
        
        # Explanation
        st.subheader("Detailed Summary")
        st.write(f"""
        - 🗣️ **Speech Detected**: Yes
        - 📝 **Transcription Excerpt**: `{transcription[:100]}{'...' if len(transcription) > 100 else ''}`
        - 🎯 **Accent Classification**: {accent}
        - ✅ **Confidence**: {'Strong' if confidence > 75 else 'Moderate' if confidence > 50 else 'Weak'}
        - ⏱️ **Audio Processed**: {duration_seconds:.1f}s
        """)

st.caption("Built with ❤️ using Python, SpeechBrain, and Streamlit")