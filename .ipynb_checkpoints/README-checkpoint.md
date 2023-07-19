# Voice cloning Model - README
This README file provides instructions on how to run the Tacotron 2 model for text-to-speech synthesis and how to evaluate its performance.

# Prerequisites
# Before running the model, make sure you have the following dependencies installed:

1) Python 3
2) PyTorch
3) torchaudio
4) scipy

# You can install the required Python packages using the following command:

# pip install torch torchaudio scipy

# Running the Model
# Import the necessary libraries:

import torch
import torchaudio
from scipy.io.wavfile import write
from tacotron2.text import text_to_sequence

# Load the Tacotron 2 and WaveGlow models:

# Load Tacotron 2 model on GPU
tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2').cuda()

# Load WaveGlow model on GPU
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow').cuda()

# Set the device to use (GPU if available):

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tacotron2 = tacotron2.to(device)
waveglow = waveglow.to(device)

# Set the text input you want to Provide:

text = "I am working on a task given by openinapp"

# Set the cleaner names (in this case, 'english_cleaners'):

cleaner_names = ['english_cleaners']

# Convert the text to sequence:

sequence = text_to_sequence(text, cleaner_names)


# Convert the text to a mel spectrogram:

sequence = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
input_lengths = torch.tensor([sequence.size(1)], device=device)
mel_outputs, mel_lengths, _ = tacotron2.infer(sequence, input_lengths=input_lengths)


# Combine the waveform using WaveGlow:

with torch.no_grad():
    audio = waveglow.infer(mel_outputs)


# Normalize the audio waveform:

audio = audio.squeeze().cpu().numpy()
audio /= audio.max()


# Save the audio as a WAV file:

output_path = 'output.wav'
write(output_path, 22050, audio)
