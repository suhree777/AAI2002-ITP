import os
import json
import ast
import pretty_midi
import numpy as np
from tensorflow.keras.models import load_model
from midi2audio import FluidSynth
import tkinter as tk
from tkinter import ttk

"""

# Dictionary to store each instrument's model
models = {}

# Load each model based on the instrument name
for instrument_name in instrument_list:
    model_path = os.path.join('ym2413_project_bt/model_folder_s', f"{instrument_name}_model.keras")
    try:
        models[instrument_name] = load_model(model_path)
        print(f"Loaded model for {instrument_name}")
    except Exception as e:
        print(f"Failed to load model for {instrument_name}: {str(e)}")
        

def generate_instrument_sequence(model, initial_sequence, sequence_length):
    sequence = initial_sequence.copy()
    for _ in range(sequence_length):
        predicted = model.predict(np.array([sequence]))[-1]
        next_value = np.argmax(predicted)
        sequence.append(next_value)
        sequence = sequence[1:]  # Slide the window
    return sequence

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        return json.load(f)

def convert_tokens_to_features(tokens, vocab):
    return [ast.literal_eval(vocab[str(token)]) for token in tokens]

def main(instruments, feature_vector):
    base_path = 'ym2413_project_bt/model_folder_s'
    vocab_base_path = 'ym2413_project_bt/3_processed_feature_limited/instrument_vocabs'
    sequence_length = 50  # Desired sequence length

    all_instruments_features = {}  # Dictionary to hold features for all instruments

    for instrument_name in instruments:
        model_path = os.path.join(base_path, f"{instrument_name}_model.keras")
        vocab_path = os.path.join(vocab_base_path, f"{instrument_name}_vocab.json")

        model = load_model(model_path)
        vocab = load_vocab(vocab_path)

        generated_tokens = generate_instrument_sequence(model, feature_vector, sequence_length)
        reverse_vocab = {v: k for k, v in vocab.items()}
        features = convert_tokens_to_features(generated_tokens, reverse_vocab)

        all_instruments_features[instrument_name] = features  # Store features by instrument name

    create_combined_midi(all_instruments_features)

def create_combined_midi(instrument_features):
    midi_file = pretty_midi.PrettyMIDI()

    for instrument_name, features in instrument_features.items():
        # Get the MIDI program number for standard instrument names, default to Acoustic Grand Piano
        program = pretty_midi.instrument_name_to_program(instrument_name if instrument_name in pretty_midi.instrument_name_to_program else "Acoustic Grand Piano")
        instrument = pretty_midi.Instrument(program=program)

        current_time = 0  # Start time for the first note
        for feature in features:
            note = pretty_midi.Note(
                velocity=int(feature['velocity']),
                pitch=int(feature['pitch']),
                start=current_time,
                end=current_time + float(feature['duration'])
            )
            instrument.notes.append(note)
            current_time += float(feature['duration'])  # Update start time for the next note

        midi_file.instruments.append(instrument)

    midi_file.write('combined_output.mid')  # Save the MIDI file
        
"""

instrument_vocab_path = 'ym2413_project_bt/3_processed_feature_limited/instrument_list.json'
with open(instrument_vocab_path, 'r') as f:
    instrument_list = json.load(f)

def load_all_models_and_vocabs():
    models = {}
    vocabs = {}
    reverse_vocabs = {}
    model_folder_path = 'ym2413_project_bt/model_folder_s'
    vocab_folder_path = 'ym2413_project_bt/3_processed_feature_limited/instrument_vocabs'

    for instrument_name in instrument_list:
        # Load each model
        model_path = os.path.join(model_folder_path, f"{instrument_name}_model.keras")
        vocab_path = os.path.join(vocab_folder_path, f"{instrument_name}_vocab.json")
        
        try:
            # Load the model
            models[instrument_name] = load_model(model_path)
            print(f"Loaded model for {instrument_name}")

            # Load and reverse the vocabulary
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
                vocabs[instrument_name] = vocab  # Store the forward vocab
                # Create and store the reverse vocab
                reverse_vocabs[instrument_name] = {v: k for k, v in vocab.items()}

            print(f"Successfully loaded model and vocab for {instrument_name}")
            # print(f"Sample of vocab for {instrument_name} contains {list(vocabs.items())[:1]}")
            # print(f"Sample of reverse vocab for {instrument_name} contains {list(reverse_vocabs.items())[:1]}")
        except Exception as e:
            print(f"Failed to load resources for {instrument_name}: {str(e)}")
    return models, vocabs, reverse_vocabs

models, vocabs, reverse_vocabs = load_all_models_and_vocabs()

# Create the root window
root = tk.Tk()
root.title("Music Generator")

moods = ['happy', 'angry', 'sad', 'relaxed']

# Mood selection
mood_label = tk.Label(root, text="Select Mood:")
mood_label.pack(pady=20)

mood_vars = {}
for mood in moods:
    var = tk.BooleanVar()
    chk = tk.Checkbutton(root, text=mood, variable=var)
    chk.pack(anchor=tk.W)
    mood_vars[mood] = var

# Instrument indicators
instru_label = tk.Label(root, text="Select Instruments:")
instru_label.pack(pady=20)

instrument_vars = {}
for instrument in instrument_list:
    var = tk.BooleanVar()
    chk = tk.Checkbutton(root, text=instrument, variable=var)
    chk.pack(anchor=tk.W)
    instrument_vars[instrument] = var

# Function to handle button click
def on_submit():
    active_moods = [mood for mood, var in mood_vars.items() if var.get()]
    active_instruments = [instr for instr, var in instrument_vars.items() if var.get()]

    # Generate binary representations
    mood_binary = [1 if mood in active_moods else 0 for mood in moods]
    instrument_binary = [1 if instrument in active_instruments else 0 for instrument in instrument_list]

    feature_vector = mood_binary + instrument_binary

    # Print selected moods, instruments, and the binary feature vector
    print("Selected Moods:", active_moods)
    print("Active Instruments:", active_instruments)
    print("Binary Representation:", feature_vector)
# Submit button
submit_button = tk.Button(root, text="Generate", command=on_submit)
submit_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()