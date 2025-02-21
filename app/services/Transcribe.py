import streamlit as st
import numpy as np
import whisper
import tempfile
import logging
import os
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import librosa


# Load the Whisper model (you can choose different sizes: "tiny", "base", "small", "medium", "large")
#@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def save_audio_to_file(audio_file):
    # Get the suffix from the uploaded file if available, otherwise use .wav
    file_suffix = Path(audio_file.name).suffix if audio_file.name else '.wav'
    
    # Create a temporary file with the same suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_audio:
        # Get the full path
        temp_path = Path(temp_audio.name)
        # Write the audio bytes to the temporary file
        temp_audio.write(audio_file.read())
        # Ensure the file is written to disk
        temp_audio.flush()
        os.fsync(temp_audio.fileno())
        
        return str(temp_path.resolve())

def transcribe_audio(audio:UploadFile = File(...),sampling_rate=16000):
    #Record audio using Streamlit's audio input
    #audio = st.file_uploader("Upload a voice message", type=["wav", "mp3", "m4a", "flac", "ogg"])
    #audio = st.audio_input("Record a voice message")
    
    if audio.filename is not None:
        # Display file information for debugging
        #st.write(f"Uploaded file name: {audio.name}")
        #st.write(f"Uploaded file type: {audio.type}")
        #st.write(f"Uploaded file size: {audio.size} bytes")
        
        # Display a loading message
        #with st.spinner("Transcribing audio..."):
        # Load the model
        model = load_whisper_model()
        
        try:
            # Save the audio data to a temporary file
            #audio_content = audio.file
            audio_file_path = save_audio_to_file(audio.file)
            
            # Verify file exists and show path for debugging
            #st.write(f"Temporary file path: {audio_file_path}")
            #st.write(f"File exists: {os.path.exists(audio_file_path)}")
            
            # Load the audio file using librosa to ensure correct format and sampling rate
            audio_array, sr = librosa.load(audio_file_path, sr=None)  # sr=None keeps original sample rate
            
            # Resample if necessary
            if sr != sampling_rate:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sampling_rate)
            
            # Transcribe the audio using Whisper
            results = model.transcribe(audio_array)
            
            # Display the transcription
            #st.write("Transcription:")
            #st.write(result["text"])

            
        except Exception as e:
            logging.info(f"Error transcribing audio: {str(e)}")
            # Print more detailed error information
            import traceback
            logging.info(f"Detailed error: {traceback.format_exc()}")
            results = {"text": "Transcription failed"}
            return results
        
        finally:
            # Clean up the temporary file if it was created
            try:
                if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
            except Exception as e:
                logging.info(f"Error cleaning up temporary file: {str(e)}")

        return results


