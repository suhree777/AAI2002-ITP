import os
import json
import torch
import torch.nn as nn
import numpy as np
import pretty_midi
import random
from midi2audio import FluidSynth
import joblib

# Load the trained model and scaler
model_path = 'ym2413_project_bt/bilstm_model.pth'
scaler_path = 'scaler.pkl'  # Ensure the scaler path is correct
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the AttentionBiLSTM model
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(AttentionBiLSTM, self).__init__()
        self.layers = nn.ModuleList()
        # Layers setup
        for i in range(len(hidden_sizes)):
            input_dim = input_size if i == 0 else hidden_sizes[i-1] * 2
            self.layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True, bidirectional=True))
        last_hidden_size = hidden_sizes[-1] * 2
        self.attention_layer = nn.Sequential(
            nn.Linear(last_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Linear(last_hidden_size, num_classes)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output, _ = layer(output)
        attention_weights = torch.nn.functional.softmax(self.attention_layer(output), dim=1)
        context_vector = torch.sum(attention_weights * output, dim=1)
        out = self.fc(context_vector)
        return out

model = AttentionBiLSTM(
    input_size=4,  # Assuming the number of features is 4
    hidden_sizes=[64, 128, 64],
    num_classes=len(np.unique(y))  # Ensure the number of classes is set correctly
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the scaler
scaler = joblib.load(scaler_path)

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
        scaled_features = output.cpu().numpy()[0]
        features = scaler.inverse_transform(scaled_features)
    
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
