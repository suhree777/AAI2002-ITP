import os
import json
import torch
import torch.nn as nn
import numpy as np
import pretty_midi
import random
from midi2audio import FluidSynth

# Define the BiLSTM model (same as the training script)
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load the trained model
model_path = 'ym2413_project_bt/bilstm_model.pth'
input_size = 4  # Number of features (pitch, velocity, duration, tempo)
hidden_size = 64
num_layers = 2
num_classes = 4  # Use the same number of classes as in the training script
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to generate input features from a random JSON file
def generate_input_features_from_json(json_folder, mood_label, sequence_length):
    mood_map = {
        'happy': 'Q1_happy',
        'angry': 'Q2_angry',
        'sad': 'Q3_sad',
        'relaxed': 'Q4_relaxed'
    }
    
    mood_folder = mood_map.get(mood_label)
    if not mood_folder:
        raise ValueError(f"Invalid mood label: {mood_label}")
    
    mood_folder_path = os.path.join(json_folder, mood_folder)
    json_files = [file for file in os.listdir(mood_folder_path) if file.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in folder: {mood_folder_path}")

    selected_file = random.choice(json_files)
    print(f'Selected file is {selected_file}')
    with open(os.path.join(mood_folder_path, selected_file), 'r') as file:
        data = json.load(file)
    
    features = []
    for instrument, instrument_features in data['instruments'].items():
        instrument_data = np.array([
            instrument_features['pitch'], 
            instrument_features['velocity'], 
            instrument_features['duration'], 
            instrument_features['tempo']
        ]).T  # Transpose to get features in the right shape
        features.append(instrument_data)
    
    combined_features = np.concatenate(features, axis=0)
    
    # If the combined features are shorter than sequence_length, pad with zeros
    if combined_features.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - combined_features.shape[0], 4))
        combined_features = np.vstack((combined_features, padding))
    
    # If the combined features are longer than sequence_length, truncate
    combined_features = combined_features[:sequence_length]
    
    return torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)

# Generate music
def generate_music(json_folder, mood_label, output_midi_path, sequence_length):
    input_features = generate_input_features_from_json(json_folder, mood_label, sequence_length)
    
    with torch.no_grad():
        output = model(input_features)
        predicted_mood = torch.sigmoid(output).cpu().numpy()[0]
        print(f'Predicted mood scores: {predicted_mood}')
    
    # Use input features to generate a MIDI file
    features = input_features.cpu().numpy()[0]

    # Generate MIDI file
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for i, feature in enumerate(features):
        pitch = int(feature[0])
        velocity = int(feature[1])
        start_time = i * (feature[2] if feature[2] > 0 else 1.0)
        end_time = start_time + (feature[2] if feature[2] > 0 else 1.0)
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_midi_path)
    print(f'Generated MIDI saved to {output_midi_path}')

    # Convert MIDI to audio using the Chiptune Soundfont
    fs = FluidSynth(soundfont_path)
    fs.midi_to_audio(output_midi_path, output_audio_path)
    print(f'Converted audio saved to {output_audio_path}')

# Parameters
json_folder = 'ym2413_project_bt/feature_output'
mood_label = 'sad'  # Example mood label
output_midi_path = 'ym2413_project_bt/generated_music.mid'
output_audio_path = 'ym2413_project_bt/generated_chiptune_music.wav'
soundfont_path = 'ym2413_project_bt/chiptune_soundfont_4.0.sf2'  # Update with the path to your Soundfont
sequence_length = 50  # Length of the generated sequence

# Generate music based on the trained model
generate_music(json_folder, mood_label, output_midi_path, sequence_length)
