import os
import json
import torch
import torch.nn as nn
import numpy as np
import pretty_midi
import random
from midi2audio import FluidSynth
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the AttentionBiLSTM model
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(AttentionBiLSTM, self).__init__()
        self.layers = nn.ModuleList()
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

# Load the trained model and scaler
model_path = 'ym2413_project_bt/bilstm_model_91.pth'
scaler_path = 'ym2413_project_bt/scaler.pkl'  # Ensure the scaler path is correct
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AttentionBiLSTM(
    input_size=4,  # Number of features for each event
    hidden_sizes=[64, 64, 64, 64],  # Match hidden sizes as per the training script
    num_classes=4  # Ensure the number of classes matches what was used during training
).to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

# Load the scaler
scaler = joblib.load(scaler_path)

def load_seed_features(json_folder, mood_label, device):
    mood_map = {
        'happy': 'Q1_happy',
        'angry': 'Q2_angry',
        'sad': 'Q3_sad',
        'relaxed': 'Q4_relaxed'
    }
    mood_folder = mood_map[mood_label]
    full_path = os.path.join(json_folder, mood_folder)
    files = os.listdir(full_path)
    selected_file = random.choice(files)
    file_path = os.path.join(full_path, selected_file)
    print(f"File selected is {selected_file}")
    with open(file_path, 'r') as f:
        data = json.load(f)
        all_features = []
        for instrument, features in data['instruments'].items():
            instrument_features = []
            for event in features:
                instrument_features.append([
                    event['pitch'],
                    event['velocity'],
                    event['duration'],
                    event['tempo']
                ])
            if instrument_features:
                instrument_features_array = np.array(instrument_features)
                all_features.append(instrument_features_array)

    combined_features = np.concatenate(all_features, axis=0) if all_features else None
    return torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Load the scaler
scaler = joblib.load(scaler_path)

def generate_music(model, device, scaler, input_features, num_steps):
    model.eval()
    generated_sequence = []

    with torch.no_grad():
        # Ensure input features start with the correct shape [1, seq_len, feature_size] and are on the correct device
        current_input = input_features.to(device)

        for _ in range(num_steps):
            output = model(current_input)
            # Get the last step output as the new feature to add to the sequence
            predicted_features = output.unsqueeze(1)  # Ensure dimensionality is correct
            generated_sequence.append(predicted_features.squeeze().cpu().numpy())
            # Concatenate along the sequence dimension for next input
            current_input = torch.cat((current_input, predicted_features), dim=1)

    # Convert generated sequence to numpy array and reverse scale
    generated_sequence = np.array(generated_sequence)
    # Reshape for scaler if necessary
    generated_sequence = generated_sequence.reshape(-1, generated_sequence.shape[-1])
    reversed_scaled_sequence = scaler.inverse_transform(generated_sequence)
    reversed_scaled_sequence = reversed_scaled_sequence.reshape(-1, num_steps, generated_sequence.shape[-1])  # Reshape back to original

    return reversed_scaled_sequence

# User input for mood
user_mood = "happy" # input("Enter a mood (happy, angry, sad, relaxed): ")

# Select and load seed input
json_folder = 'ym2413_project_bt/feature_output'
seed_features = load_seed_features(json_folder, user_mood, device)  # Include the device argument here

# Generate music
generated_music_features = generate_music(model, device, scaler, seed_features, num_steps=50)

# Convert generated features to MIDI
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
start_time = 0
for feature in generated_music_features:
    print("Feature array: \n", feature)  # Print the entire feature array
    print("Type of feature:", type(feature))  # Check the type of feature
    print("Shape of feature:", feature.shape if isinstance(feature, np.ndarray) else "Not an array")  # Print shape if it's an array
    
    # Ensure pitch and velocity are within MIDI standards
    pitch = min(max(int(feature[0]), 0), 127)
    velocity = min(max(int(feature[1]), 0), 127)
    duration = feature[2]  # duration for readability in prints
    end_time = start_time + duration

    print("Pitch:", pitch)        # Print the computed pitch
    print("Velocity:", velocity)  # Print the computed velocity
    print("Start time:", start_time)  # Print the start time for the note
    print("End time:", end_time)  # Print the end time for the note

    note = pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=start_time,
        end=end_time
    )
    instrument.notes.append(note)
    start_time = end_time  # Increment start time by duration
midi.instruments.append(instrument)
midi.write('generated_music.mid')

# Convert MIDI to audio using FluidSynth
soundfont_path = 'ym2413_project_bt/chiptune_soundfont_4.0.sf2'
output_audio_path = 'generated_music.wav'
fs = FluidSynth(soundfont_path)
fs.midi_to_audio('generated_music.mid', output_audio_path)

print(f"Generated music saved as 'generated_music.wav'")