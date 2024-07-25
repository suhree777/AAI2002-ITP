import os
import json
import ast
import random
import pretty_midi
import torch
from torch import nn
from datetime import datetime
from midi2audio import FluidSynth
import tkinter as tk

directory_path = 'ym2413_project_bt/3_processed_freq/'
summary_path = os.path.join(directory_path, 'summary.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open(summary_path, 'r') as f:
    data = json.load(f)
    instrument_list = data['top_instrument_names']
    moods = data['mood_labels']
    sample_rate = data['desired_sample_rate']
    size_multiplyer = data['size_multiplyer']
    embedding_dim = data['embedding_dim']

vocab_file_path = os.path.join(directory_path, 'event_vocab.json')
with open(vocab_file_path, 'r') as f:
    vocab = json.load(f)

token_size = len(vocab) + len(instrument_list) + len(moods)
window_length = sample_rate * size_multiplyer # Sliding window size per instrment
embedding_dim = embedding_dim

class MusicModel(nn.Module):
    def __init__(self, token_size, embedding_dim):
        super(MusicModel, self).__init__()
        self.embedding = nn.Embedding(token_size, embedding_dim).to(device)
        self.lstm1 = nn.LSTM(embedding_dim, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.15)
        """self.lstm2 = nn.LSTM(256, 256, batch_first=True)
        self.dropout2 = nn.Dropout(0.15)"""
        self.dense = nn.Linear(256, token_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)  # x has shape [batch, sequence_length, 256]
        x = self.dropout1(x)
        """x, _ = self.lstm2(x)  # x has shape [batch, sequence_length, 256]
        x = self.dropout2(x)"""
        x = self.dense(x[:, -1, :])  # Selecting the last timestep output for each sequence
        return x

def load_all_models_and_vocabs(instrument_list, vocab):
    models = {}
    model_folder = 'ym2413_project_bt/model_folder_freq'
    reverse_vocab = {value: key for key, value in vocab.items()}

    print(f"Now loading models from {model_folder}")
    print(f"Successfully loaded and reverse vocab")

    for instrument_name in instrument_list:
        model_path = os.path.join(model_folder, f"{instrument_name}_model.pt")
        try:
            model = MusicModel(token_size, embedding_dim).to(device)  # Create an instance and move to device
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            models[instrument_name] = model
            print(f"Loaded model for {instrument_name}")
        except Exception as e:
            print(f"Failed to load model for {instrument_name}: {str(e)}")
    return models, reverse_vocab

def convert_predictions_to_midi(predictions, reverse_vocab, sample_rate):
    predictions_by_instrument = list(zip(*predictions))
    midi = pretty_midi.PrettyMIDI()
    time_increment = 1 / sample_rate
    overlap_factor = 1.8  # Extend each note's duration by 80%

    for instrument_index, instrument_predictions in enumerate(predictions_by_instrument):
        instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
        current_time = 0
        for token in instrument_predictions:
            if token in reverse_vocab:
                note_attributes = ast.literal_eval(reverse_vocab[token])
                pitch = note_attributes[0][1]
                velocity = note_attributes[1][1]
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=current_time,
                    end=current_time + time_increment * overlap_factor
                )
                instrument.notes.append(note)
                current_time += time_increment
        midi.instruments.append(instrument)
    return midi

def midi_to_audio(midi_file_path, sound_font, output_audio_path):
    fs = FluidSynth(sound_font)
    fs.midi_to_audio(midi_file_path, output_audio_path)
    print(f'Converted audio saved to {output_audio_path}')

def update_sequence(input_sample, prediction, instrument_index, window_length):
    start_idx = len(moods) + len(instrument_list) + instrument_index * window_length
    end_idx = min(start_idx + window_length, len(input_sample))

    temp_input_sample = input_sample.clone()
    input_sample[start_idx + 1:end_idx] = temp_input_sample[start_idx:end_idx - 1]

    if end_idx - 1 < len(input_sample):
        input_sample[end_idx - 1] = prediction

def generate_sequence(models, feature_vector, num_generate, window_length):
    # Convert feature_vector to a tensor and move it to the device
    input_sample = torch.tensor(feature_vector, dtype=torch.long, device=device)
    # print(f"Initial feature vector: {feature_vector}")

    predictions = []
    for step in range(num_generate):
        current_predictions = []
        # print(f"Generation step {step + 1}")

        for i, (instrument_name, model) in enumerate(models.items()):
            # Ensure the input tensor is on the same device as the model
            input_tensor = input_sample.unsqueeze(0).to(device)  # Add batch dimension and ensure device consistency
            # print(f"Processing model for {instrument_name}")

            with torch.no_grad():
                output = model(input_tensor)
            predicted_token = output.argmax(dim=1).item()
            current_predictions.append(predicted_token)

            # print(f"Predicted token for {instrument_name}: {predicted_token}")

            # Update sequence, ensuring all operations stay on the correct device
            update_sequence(input_sample, predicted_token, i, window_length)

        predictions.append(current_predictions)
        print(f"Current predictions: {current_predictions}")

    return predictions


def get_sample_seed(primary_mood, active_instruments, mood_binary, instrument_binary, instrument_list, window_length, sample_path):
    mood_folder = os.path.join(sample_path, primary_mood)
    print(f"Looking for a sample in {mood_folder}")
    
    # Ensure the mood folder exists and has files
    if not os.path.exists(mood_folder) or not os.listdir(mood_folder):
        raise FileNotFoundError(f"No seed files found in folder: {mood_folder}")

    suitable_file_found = False
    seed_data = None

    while not suitable_file_found:
        # Select a random JSON file from the mood folder
        seed_file = random.choice(os.listdir(mood_folder))
        seed_path = os.path.join(mood_folder, seed_file)

        # Load data from the selected seed file
        with open(seed_path, 'r') as file:
            seed_data = json.load(file)

        # Check if the seed data contains at least two active instruments
        active_in_seed = [instr for instr in seed_data['instruments'] if instr in active_instruments]
        if len(active_in_seed) >= 2:
            suitable_file_found = True
        else:
            print(f"Sample in {seed_file} does not meet instrument criteria, selecting another...")

    # Initialize seed values for each instrument with zeros
    seed_values = {instr: [0] * window_length for instr in instrument_list}
    # Replace zero arrays with actual seed values from file where available
    for instr, sequence in seed_data['instruments'].items():
        if instr in seed_values:
            seed_values[instr] = sequence[:window_length]

    # Construct the complete initial sequence
    instrument_seed = []
    for instr in instrument_list:
        instrument_seed.extend(seed_values[instr])

    # Combine mood vector, instrument binary, and seed values into a single initial sequence
    initial_sequence = mood_binary + instrument_binary + instrument_seed
    print(f"Completed sample seed sequence with suitable instruments: {active_in_seed}")
    return initial_sequence

models, reverse_vocab = load_all_models_and_vocabs(instrument_list, vocab)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f'ym2413_project_bt/gen_samples/'  # Path to your MIDI file
sound_font = 'ym2413_project_bt/chiptune_soundfont_4.0.sf2'  # Path to your sound font file
sample_path = 'ym2413_project_bt/sample_seed/'

def ensure_mood_exclusivity(primary_mood_var, secondary_mood_vars):
    primary_mood = primary_mood_var.get()
    for mood, var in secondary_mood_vars.items():
        if mood == primary_mood:
            var.set(False)

# Create the root window
root = tk.Tk()
root.title("Music Generator")
root.geometry(f'450x{490 + (50 * len(instrument_list))}')

prefix_length = len(moods) + len(instrument_list)  # The first 20 tokens are fixed

# Primary Mood Selection
primary_mood_label = tk.Label(root, text="Select Primary Mood:")
primary_mood_label.pack(pady=20)

default_mood = moods[0]  # or any specific mood like "Happy"
primary_mood_var = tk.StringVar(value=default_mood)
for mood in moods:
    rb = tk.Radiobutton(root, text=mood, variable=primary_mood_var, value=mood,
                        command=lambda: ensure_mood_exclusivity(primary_mood_var, secondary_mood_vars))
    rb.pack(anchor=tk.W)

# Secondary Mood Selection
secondary_mood_label = tk.Label(root, text="Select Secondary Moods:")
secondary_mood_label.pack(pady=10)

secondary_mood_vars = {}
for mood in moods:
    var = tk.BooleanVar()
    chk = tk.Checkbutton(root, text=mood, variable=var,
                         command=lambda m=mood: ensure_mood_exclusivity(primary_mood_var, secondary_mood_vars))
    chk.pack(anchor=tk.W)
    secondary_mood_vars[mood] = var

# Instrument indicators
instru_label = tk.Label(root, text="Select Instruments:")
instru_label.pack(pady=20)

instrument_vars = {}
for instrument in instrument_list:
    var = tk.BooleanVar()
    chk = tk.Checkbutton(root, text=instrument, variable=var)
    chk.pack(anchor=tk.W)
    instrument_vars[instrument] = var

# Option for seed selection
seed_label = tk.Label(root, text="Initial Sequence Source:")
seed_label.pack(pady=10)

seed_var = tk.StringVar(value="sample")  # Default to zero padding
seed_choices = {"Zero Padding": "zero", "Sample from Dataset": "sample"}
for text, mode in seed_choices.items():
    rb = tk.Radiobutton(root, text=text, variable=seed_var, value=mode)
    rb.pack(anchor=tk.W)

# Add a label and entry for duration input
duration_label = tk.Label(root, text="Enter duration in seconds (max 60s):")
duration_label.pack(pady=10)

duration_var = tk.StringVar()
duration_entry = tk.Entry(root, textvariable=duration_var)
duration_entry.pack()

# Set a default value
duration_var.set("10")

# Validate and use the input
def validate_duration():
    try:
        duration = int(duration_var.get())
        if duration <= 0 or duration > 60:
            raise ValueError("Duration must be between 1 and 60 seconds.")
        return duration
    except ValueError as e:
        print("Invalid input for duration:", e)
        return None

# Function to handle button click
def on_submit():
    primary_mood = primary_mood_var.get()
    active_secondary_moods = [mood for mood, var in secondary_mood_vars.items() if var.get()]
    combined_moods = [primary_mood] + [mood for mood in active_secondary_moods if mood != primary_mood]
    active_instruments = [instr for instr, var in instrument_vars.items() if var.get()]

    # Generate binary representations
    mood_binary = [1 if mood in combined_moods else 0 for mood in moods]
    instrument_binary = [1 if instrument in active_instruments else 0 for instrument in instrument_list]

    seed_choice = seed_var.get()
    if seed_choice == "zero":
        seed_value = [0] * (window_length * len(instrument_list))  # Zero padding
        feature_vector = mood_binary + instrument_binary + seed_value
    else:
        feature_vector = get_sample_seed(primary_mood, active_instruments, mood_binary, instrument_binary, instrument_list, window_length, sample_path)

    duration = validate_duration()
    if duration is None:  # If validation fails, do not proceed
        return

    num_generate = int(duration * sample_rate)
    
    # Print selected moods, instruments, and the binary feature vector
    print("Selected Moods:", combined_moods)
    print("Active Instruments:", active_instruments)
    print("Binary Representation:", feature_vector)

    initial_sequence = feature_vector.copy()
    generated_tokens = generate_sequence(models, initial_sequence, num_generate, window_length)
    midi = convert_predictions_to_midi(generated_tokens, reverse_vocab, sample_rate)
    midi_path = os.path.join(output_path, f'{primary_mood}_{current_time}_output.mid')
    audio_path = os.path.join(output_path, f'{primary_mood}_{current_time}_output.wav')
    midi.write(midi_path)  # Save the MIDI file
    midi_to_audio(midi_path, sound_font, audio_path)

# Submit button
submit_button = tk.Button(root, text="Generate", command=on_submit)
submit_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()