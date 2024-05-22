import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pretty_midi
import random
from midi2audio import FluidSynth
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
            nn.Linear(128, num_classes)  # Output size here should match the desired sequence output features
        )

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x, _ = layer(x)
            outputs.append(x)
        x = outputs[-1]  # Consider using outputs from the last LSTM layer
        # Apply attention to each time step if needed
        x = self.attention_layer(x)
        return x

def load_instrument_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        return json.load(file)

def load_reverse_instrument_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
    # Reverse the dictionary to map indices to instrument names
    return {v: k for k, v in vocab.items()}

# Load the trained model and scaler
model_path = 'ym2413_project_bt/bilstm_model_2_73.pth'
scaler_path = 'ym2413_project_bt/scaler_2_73.pkl'  # Ensure the scaler path is correct

model = AttentionBiLSTM(
    input_size=5,  # Number of features for each event
    hidden_sizes=[64, 64, 64, 64],  # Match hidden sizes as per the training script
    num_classes=4  # Ensure the number of classes matches what was used during training
).to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

# Load the scaler
scaler = joblib.load(scaler_path)
instrument_vocab = load_instrument_vocab('ym2413_project_bt/feature_extracted/instrument_vocab.json')
reverse_vocab = load_reverse_instrument_vocab('ym2413_project_bt/feature_extracted/instrument_vocab.json')

def load_seed_features(json_folder, mood_label, device, instrument_vocab):
    try:
        mood_map = {
            'happy': 'Q1_happy',
            'angry': 'Q2_angry',
            'sad': 'Q3_sad',
            'relaxed': 'Q4_relaxed'
        }
        mood_folder = mood_map[mood_label]
        full_path = os.path.join(json_folder, mood_folder)
        files = os.listdir(full_path)
        if not files:
            raise ValueError(f"No files found in directory {full_path}")
        
        selected_file = random.choice(files)
        file_path = os.path.join(full_path, selected_file)
        print(f"File selected is {selected_file}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_features = []
        for instrument, features in data['instruments'].items():
            instrument_idx = instrument_vocab.get(instrument, -1)  # Retrieve instrument index
            for event in features:
                # Include instrument index in the feature vector
                instrument_features = [instrument_idx] + [event['pitch'], event['velocity'], event['duration'], event['tempo']]
                all_features.append(instrument_features)
        
        if not all_features:
            raise ValueError("No features found in the selected file.")
        
        combined_features = np.array(all_features)
        combined_features = scaler.transform(combined_features)  # Apply the same scaler as used in training
        return torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    except Exception as e:
        print(f"Error loading seed features: {e}")
        return None

def generate_music(model, device, scaler, input_features, num_steps):
    model.eval()
    generated_sequence = []

    with torch.no_grad():
        current_input = input_features.to(device)
        print(f"current_input value is {current_input}")
        for _ in range(num_steps):
            output = model(current_input)
            print(f"output value is {output}")
            # Directly use output as predicted features (no need to slice)
            predicted_features = output.unsqueeze(1)  # Maintain batch and sequence dimension consistency

            generated_sequence.append(predicted_features.squeeze().cpu().numpy())
            # Concatenate along the sequence dimension for next input
            current_input = torch.cat((current_input, predicted_features), dim=1)

    # Convert generated sequence to numpy array and reverse scale
    generated_sequence = np.array(generated_sequence)
    generated_sequence = generated_sequence.reshape(-1, generated_sequence.shape[-1])
    reversed_scaled_sequence = scaler.inverse_transform(generated_sequence)
    reversed_scaled_sequence = reversed_scaled_sequence.reshape(-1, num_steps, generated_sequence.shape[-1])

    return reversed_scaled_sequence


# User input for mood
user_mood = "happy" # input("Enter a mood (happy, angry, sad, relaxed): ")

# Select and load seed input
json_folder = 'ym2413_project_bt/feature_extracted'
seed_features = load_seed_features(json_folder, user_mood, device, instrument_vocab)  # Include the device argument here
print(f"seed_features print out {seed_features}")

assert seed_features.shape[2] == model.layers[0].input_size, "Mismatch in feature size"

# Generate music
generated_music_features = generate_music(model, device, scaler, seed_features, num_steps=50)

# Convert generated features to MIDI
midi = pretty_midi.PrettyMIDI()
start_time = 0

for feature in generated_music_features:
    # Retrieve instrument index and find corresponding MIDI program
    instrument_index = int(feature[0])
    instrument_name = reverse_vocab.get(instrument_index, 'Acoustic Grand Piano')  # Default to piano if not found
    program = pretty_midi.instrument_name_to_program(instrument_name)
    
    # Create a new instrument object for each note or reuse if the same instrument continues
    instrument = pretty_midi.Instrument(program=program)
    
    # MIDI values
    pitch = min(max(int(feature[1]), 0), 127)
    velocity = min(max(int(feature[2]), 0), 127)
    duration = feature[3]  # Assuming feature[3] is duration
    end_time = start_time + duration

    note = pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=start_time,
        end=end_time
    )
    instrument.notes.append(note)
    midi.instruments.append(instrument)
    start_time = end_time  # Update start time for the next note

midi.write('generated_music.mid')

# Convert MIDI to audio using FluidSynth
soundfont_path = 'ym2413_project_bt/chiptune_soundfont_4.0.sf2'
output_audio_path = 'generated_music.wav'
fs = FluidSynth(soundfont_path)
fs.midi_to_audio('generated_music.mid', output_audio_path)

print(f"Generated music saved as 'generated_music.wav'")