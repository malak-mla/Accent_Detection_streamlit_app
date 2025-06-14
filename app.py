import streamlit as st
import requests
import os
import torch
from pydub import AudioSegment
import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier
from io import BytesIO
import tempfile

# Configuration
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_AUDIO_DURATION = 60000  # 60 seconds (in milliseconds)

# Initialize models (cached for performance)
@st.cache_resource
def load_accent_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/acc-identification-commonaccent_ecapa",
        savedir="pretrained_models"
    )

def download_file(url):
    """Download file from URL with size limit"""
    # Get content length first
    head_response = requests.head(url)
    if 'content-length' in head_response.headers:
        file_size = int(head_response.headers['content-length'])
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large ({file_size/1024/1024:.1f}MB > {MAX_FILE_SIZE/1024/1024}MB limit)")
    
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file (HTTP {response.status_code})")
    
    # Check extension
    ext = url.split('.')[-1].lower()
    if ext not in ['mp4', 'mov', 'avi', 'webm', 'mkv']:
        ext = 'mp4'  # Default extension
    
    # Download with size limit
    downloaded_bytes = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_bytes += len(chunk)
            if downloaded_bytes > MAX_FILE_SIZE:
                raise ValueError(f"File exceeds {MAX_FILE_SIZE/1024/1024}MB size limit")
            tmp_file.write(chunk)
        return tmp_file.name

def extract_audio(video_path):
    """Extract audio from video with duration limit"""
    try:
        audio = AudioSegment.from_file(video_path)
    except:
        # Try different formats
        try:
            audio = AudioSegment.from_file(video_path, format="mp4")
        except Exception as e:
            raise ValueError(f"Unsupported video format: {str(e)}")
    
    # Convert to mono and resample
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Truncate to max duration
    if len(audio) > MAX_AUDIO_DURATION:
        audio = audio[:MAX_AUDIO_DURATION]
        st.warning(f"Audio truncated to {MAX_AUDIO_DURATION/1000} seconds")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
        audio.export(tmp_audio.name, format="wav")
        return tmp_audio.name

def transcribe_audio(audio_path):
    """Transcribe audio and return text"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            st.error(f"Speech recognition error: {str(e)}")
            return ""

def classify_accent(audio_path, model):
    """Classify accent and return results"""
    try:
        # Get classification
        out_prob, score, index, text_lab = model.classify_file(audio_path)
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
st.write(f"""
Upload a public video URL to analyze the speaker's English accent.
**Limits:** Max {MAX_FILE_SIZE/1024/1024}MB file, {MAX_AUDIO_DURATION/1000}s audio analyzed
""")

video_url = st.text_input("Enter video URL:", placeholder="https://example.com/video.mp4")

if st.button("Analyze Accent"):
    if not video_url:
        st.warning("Please enter a valid video URL")
    else:
        with st.spinner("Processing..."):
            try:
                # Step 1: Download video
                video_path = download_file(video_url)
                
                # Step 2: Extract audio
                audio_path = extract_audio(video_path)
                
                # Audio preview
                st.subheader("Extracted Audio")
                st.audio(audio_path, format='audio/wav')
                
                # Step 3: Transcribe audio
                transcription = transcribe_audio(audio_path)
                
                if not transcription:
                    st.error("No English speech detected in the first 60 seconds")
                    st.stop()
                
                # Step 4: Classify accent
                model = load_accent_model()
                accent, confidence = classify_accent(audio_path, model)
                
                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                col1.metric("Detected Accent", accent)
                col2.metric("Confidence Score", f"{confidence:.1f}%")
                
                st.progress(int(confidence))
                
                # Explanation
                st.subheader("Detailed Summary")
                st.write(f"""
                - 🗣️ **Speech Detected**: {'Yes' if transcription else 'No'}
                - 📝 **Transcription Excerpt**: `{transcription[:100]}{'...' if len(transcription) > 100 else ''}`
                - 🎯 **Accent Classification**: {accent}
                - ✅ **Confidence**: {'Strong' if confidence > 75 else 'Moderate' if confidence > 50 else 'Weak'}
                - ⏱️ **Audio Processed**: {min(len(AudioSegment.from_file(audio_path)), MAX_AUDIO_DURATION)/1000}s
                """)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Clean up temporary files
                for path in [video_path, audio_path]:
                    if path and os.path.exists(path):
                        try:
                            os.unlink(path)
                        except:
                            pass

st.caption("Built with ❤️ using Python, SpeechBrain, and Streamlit")