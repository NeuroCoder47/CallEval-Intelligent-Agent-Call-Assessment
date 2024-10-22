import streamlit as st
import os
import subprocess
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

# Function to convert m4a to wav
def convert_m4a_to_wav(input_file, output_file):
    ffmpeg_path = r"C:\FFmpeg\bin"  # Adjust this to your actual FFmpeg path
    os.environ['PATH'] += os.pathsep + ffmpeg_path

    command = ['ffmpeg', '-i', input_file, output_file]
    try:
        subprocess.run(command, check=True)
        st.success(f"Successfully converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred during conversion: {e}")
    except FileNotFoundError:
        st.error("FFmpeg not found. Please check the installation and path.")

# Function to transcribe audio
def transcribe_audio(audio_file):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    # Load the audio file
    audio_input, original_sampling_rate = torchaudio.load(audio_file)

    # Resample the audio to 16,000 Hz if it's not already in that sampling rate
    target_sampling_rate = 16000
    if original_sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
        audio_input = resampler(audio_input)

    # Whisper expects mono audio, so convert if necessary
    if audio_input.shape[0] > 1:
        audio_input = audio_input.mean(dim=0, keepdim=True)

    # Split into manageable chunks of 30 seconds (if needed)
    chunk_length = 30  # seconds
    num_chunks = audio_input.size(1) // (chunk_length * target_sampling_rate)
    transcriptions = []

    for i in range(num_chunks + 1):
        start = i * chunk_length * target_sampling_rate
        end = start + chunk_length * target_sampling_rate
        chunk = audio_input[:, start:end]

        if chunk.size(1) == 0:
            break

        # Process and get the transcription for each chunk
        input_features = processor(chunk.squeeze().numpy(), sampling_rate=target_sampling_rate, return_tensors="pt").input_features

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription[0])

    # Join all chunk transcriptions together
    return ' '.join(transcriptions)


# Function to analyze transcription using LLM
def analyze_transcription(transcription_text):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    input_text = (
        f"{transcription_text}\n\n"
        "Answer the following questions based on the conversation:\n"
        "1. Did the agent follow the policies? (yes/no)\n"
        "2. Was the agent effective? (yes/no/nil)\n"
        "3. Did the borrower promise to pay? (yes/no/nil)\n"
        "4. What's the next step? (ans/nil)\n"
    )

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(
        input_ids, 
        max_new_tokens=100,
        num_beams=5,
        early_stopping=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("Agent Call Assessment App")

# Upload the call recording
st.header("Upload the call recording here")
audio = st.file_uploader("Please upload the call recording", type=["mp3", "m4a"])

# Convert and transcribe the audio
if audio is not None:
    input_file_path = f"temp_{audio.name}"
    with open(input_file_path, "wb") as f:
        f.write(audio.read())

    output_file_path = f"output_audio.wav"

    if st.button("Convert to WAV"):
        convert_m4a_to_wav(input_file_path, output_file_path)

        with open(output_file_path, "rb") as f:
            st.download_button("Download Converted WAV", f, file_name=output_file_path)

# Transcription and state management
if 'transcription' not in st.session_state:
    st.session_state['transcription'] = ''

st.subheader("Transcription of the audio")
audio1 = st.file_uploader("Please upload the WAV file you just downloaded", type=["wav"])

if audio1 is not None:
    if st.button("Start Transcription"):
        st.session_state['transcription'] = transcribe_audio(audio1)
        st.write(st.session_state['transcription'])

# Display the transcription once it's done
if st.session_state['transcription']:
    st.write("Transcription:")
    st.write(st.session_state['transcription'])

    # Button to analyze transcription
    if st.button("Analyze Transcription"):
        # Perform the analysis and store it in session state
        st.session_state['analysis_result'] = analyze_transcription(st.session_state['transcription'])

# Display the analysis results
if 'analysis_result' in st.session_state:
    st.subheader("Conversation Analysis")
    st.write("""
        1. Did the agent follow the policies?  
        2. Was the agent effective?  
        3. Did the borrower promise to pay?  
        4. What's the next step?
    """)
    st.write(f"Analysis Result:\n{st.session_state['analysis_result']}")
