import numpy as np
import librosa
import soundfile as sf
import os

folderDirectory = r'Desktop\ITP\test'
# Load the MP3 audio file (replace 'your_audio_file.mp3' with the path to your MP3 file)
audio_file = os.path.join(folderDirectory, 'Paradrizzle 192k.mp3')
audio, sample_rate = librosa.load(audio_file)

# Extract melody and harmony using Harmonic-Percussive Source Separation
harmonic, percussive = librosa.effects.hpss(audio)

# Save the separated components as new audio files
sf.write(os.path.join(folderDirectory, 'melody.mp3'), harmonic, sample_rate)
sf.write(os.path.join(folderDirectory, 'harmony.mp3'), percussive, sample_rate)