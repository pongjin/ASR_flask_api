from flask import Flask, request, jsonify
import torch
import wave
import numpy as np
import scipy.signal as signal
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import os

app = Flask(__name__)

# Define the paths for the pre-trained models and tokenizer
ASR_model = "pongjin/en_with_korean_model_large_960h"

# Load the pre-trained models and tokenizer
encoder = Wav2Vec2Processor.from_pretrained(ASR_model)
model = Wav2Vec2ForCTC.from_pretrained(ASR_model)
decoder = Wav2Vec2ProcessorWithLM.from_pretrained(ASR_model)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if the request contains a file
    audio_file = request.files['file']

    # Open the input WAV file
    with wave.open(audio_file, "rb") as wav_in:
        input_sample_rate = wav_in.getframerate()
        audio_data = wav_in.readframes(-1)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

    # Resample the audio data
    rate_ratio = 16000 / input_sample_rate
    resampled_audio = signal.resample(audio_data, int(len(audio_data) * rate_ratio))

    # Set the interval for slicing audio data (in seconds)
    interval = 10

    # Store sliced audio data
    sliced_audio_data = []

    # Slice the audio data at the given interval
    start_time = 0
    end_time = interval * 16000
    while end_time <= resampled_audio.size:
        sliced_audio_data.append(resampled_audio[start_time:end_time])
        start_time += interval * 16000
        end_time += interval * 16000

    #Process the sliced audio data using the Wav2Vec model and LM model
    total_audio = []
    for sliced in sliced_audio_data:
        # Tokenize and encode the input
        inputs = encoder(sliced, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            logits = model(inputs['input_values'].squeeze(1)).logits
        transcription = decoder.batch_decode(logits.numpy()).text

        total_audio.append(transcription[0].lower())
    
    results = ' '.join(total_audio)

    return jsonify({'transcription': results}), 200


if __name__ == '__main__':
    app.run()
