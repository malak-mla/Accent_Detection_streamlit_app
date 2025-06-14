import streamlit as st
import requests
import os
import torch
from pydub import AudioSegment
import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier
from io import BytesIO
import tempfile

# Initialize models (cached for performance)
@st.cache_resource
def load_accent_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/acc-identification-commonaccent_ecapa",
        savedir="pretrained_models"
    )

def download_file(url):
    """Download file from URL and return local path"""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file (HTTP {response.status_code})")
    
    ext = url.split('.')[-1].lower()
    if ext not in ['mp4', 'mov', 'avi', 'webm']:
        ext = 'mp4'  # Default extension
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        return tmp_file.name

def extract_audio(video_path):
    """Extract audio from video and return WAV path"""
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Resample to 16kHz
    
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
            'us': 'American'
        }
        
        return accent_map.get(accent_code, accent_code), min(confidence, 100)
    except Exception as e:
        st.error(f"Accent classification error: {str(e)}")
        return "Error", 0.0

# Streamlit UI
st.title("🎤 English Accent Detection")
st.write("""
Upload a public video URL to analyze the speaker's English accent.
Supported formats: MP4, Loom links, or direct video links.
""")

video_url = st.text_input("Enter video URL:", placeholder="https://example.com/video.mp4")

if st.button("Analyze Accent"):
    if not video_url:
        st.warning("Please enter a valid video URL")
    else:
        with st.spinner("Processing video..."):
            try:
                # Step 1: Download video
                video_path = download_file(video_url)
                
                # Step 2: Extract audio
                audio_path = extract_audio(video_path)
                st.audio(audio_path, format='audio/wav')
                
                # Step 3: Transcribe audio
                transcription = transcribe_audio(audio_path)
                
                if not transcription:
                    st.error("No English speech detected")
                    st.stop()
                
                # Step 4: Classify accent
                model = load_accent_model()
                accent, confidence = classify_accent(audio_path, model)
                
                # Display results
                st.subheader("Results")
                st.metric("Detected Accent", accent)
                st.metric("Confidence Score", f"{confidence:.1f}%")
                
                st.progress(int(confidence))
                
                # Explanation
                st.subheader("Analysis Summary")
                st.write(f"""
                - 🗣️ **Speech Detected**: {'Yes' if transcription else 'No'}
                - 📝 **Transcription Excerpt**: `{transcription[:50]}...`
                - 🎯 **Accent Classification**: {accent}
                - ✅ **Confidence**: {'Strong' if confidence > 75 else 'Moderate' if confidence > 50 else 'Weak'}
                """)
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
            finally:
                # Clean up temporary files
                for path in [video_path, audio_path]:
                    if path and os.path.exists(path):
                        os.unlink(path)

st.caption("Built with ❤️ using Python, SpeechBrain, and Streamlit")