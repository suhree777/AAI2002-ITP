import os
import json
import ast
import pretty_midi
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from midi2audio import FluidSynth
import tkinter as tk

directory_path = 'ym2413_project_bt/3_processed_freq/'
summary_path = os.path.join(directory_path, 'summary.json')
with open(summary_path, 'r') as f:
    data = json.load(f)
    instrument_list = data['top_instrument_names']
    moods = data['mood_labels']
    sample_rate = data['desired_sample_rate']

def load_all_models_and_vocabs(directory_path, instrument_list):
    models = {}
    model_folder = 'ym2413_project_bt/model_folder_freq'
    vocab_path = os.path.join(directory_path, 'event_vocab.json')
    # Load and reverse the vocabulary
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
        reverse_vocab = {value: key for key, value in vocab.items()}

    print(f"Successfully loaded and reverse vocab")
    for instrument_name in instrument_list:
        # Load each model
        model_path = os.path.join(model_folder, f"{instrument_name}_model.keras")
        
        try:
            # Load the model
            models[instrument_name] = load_model(model_path)
            print(f"Loaded model for {instrument_name}")
        except Exception as e:
            print(f"Failed to load resources for {instrument_name}: {str(e)}")
    return models, reverse_vocab

def generate_interdependent_sequence(models, initial_sequence, prefix_length, window_length, instrument_list, num_generate):
    # Assume initial_sequence is properly formatted to include prefixed features + initial instrument states
    current_sequence = list(initial_sequence)  # Copy to mutable list

    # Container for storing the generated sequences per instrument
    generated_sequences = {name: [] for name in instrument_list}

    for _ in range(num_generate):
        # Container for the next tokens predicted by each model
        next_tokens = {}

        # Generate a token for each instrument
        for name, model in models.items():
            # Extract the relevant portion of the current_sequence for this model
            input_sequence = np.array([current_sequence[-(prefix_length + window_length * len(instrument_list)):]])
            predicted_probs = model.predict(input_sequence)[0]
            next_token = np.argmax(predicted_probs)
            next_tokens[name] = next_token

        # Append the generated token to the sequence and to the individual generated sequences
        for name, token in next_tokens.items():
            current_sequence.append(token)
            generated_sequences[name].append(token)

    return generated_sequences


def parse_token(token, vocab):
    try:
        token_str = vocab[token]
        return ast.literal_eval(token_str)  # Safely evaluate string to tuple
    except SyntaxError:
        return "Invalid Token"  # Handle any tokens that might still cause issues

def clean_sequence(sequence):
    cleaned_sequence = []
    for note in sequence:
        # Extracting duration, pitch, and velocity directly by assuming the order and structure
        duration = note[0][1]
        pitch = note[1][1]
        velocity = note[3][1]
        cleaned_sequence.append((duration, pitch, velocity))
    return cleaned_sequence

def midi_to_audio(midi_file_path, sound_font, output_audio_path):
    # Create a synthesizer object with the sound font
    fs = FluidSynth(sound_font)
    fs.midi_to_audio(midi_file_path, output_audio_path)
    print(f'Converted audio saved to {output_audio_path}')

def create_midi_from_sequences(instrument_sequences, instrument_names, midi_file_path):
    midi = pretty_midi.PrettyMIDI()
    for instrument_name, sequence in zip(instrument_names, instrument_sequences):
        program = pretty_midi.instrument_name_to_program(instrument_name)
        instrument = pretty_midi.Instrument(program=program)
        current_time = 0.0
        for duration, pitch, velocity in sequence:
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=current_time,
                end=current_time + duration
            )
            instrument.notes.append(note)
            current_time += duration
        midi.instruments.append(instrument)
    midi.write(midi_file_path)

models, reverse_vocab = load_all_models_and_vocabs(directory_path, instrument_list)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
midi_file_path = f'ym2413_project_bt/gen_samples/output{current_time}.mid'  # Path to your MIDI file
sound_font = 'ym2413_project_bt/chiptune_soundfont_4.0.sf2'  # Path to your sound font file
output_audio_path = f'ym2413_project_bt/gen_samples/output{current_time}.wav'  # Path to save the output audio file

# Create the root window
root = tk.Tk()
root.title("Music Generator")
root.geometry(f'450x{450 + (30 * len(instrument_list))}')

prefix_length = len(moods) + len(instrument_list)  # The first 20 tokens are fixed
window_length = sample_rate  # The last 10 tokens slide

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

# Option for seed selection
seed_label = tk.Label(root, text="Initial Sequence Source:")
seed_label.pack(pady=10)

seed_var = tk.StringVar(value="zero")  # Default to zero padding
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
duration_var.set("30")  # Default to 30 seconds

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
    active_moods = [mood for mood, var in mood_vars.items() if var.get()]
    active_instruments = [instr for instr, var in instrument_vars.items() if var.get()]

    # Generate binary representations
    mood_binary = [1 if mood in active_moods else 0 for mood in moods]
    instrument_binary = [1 if instrument in active_instruments else 0 for instrument in instrument_list]

    seed_choice = seed_var.get()
    if seed_choice == "zero":
        seed_value = [0] * (window_length * len(instrument_list))  # Zero padding
    # else:
        # feature_vector = get_sample_seed(instrument, window_length)
    # seed_value = [0] * window_length if seed_var.get() == "zero" else get_sample_seed()

    duration = validate_duration()
    if duration is None:  # If validation fails, do not proceed
        return

    num_generate = int(duration * sample_rate)

    feature_vector = mood_binary + instrument_binary + seed_value
    # Print selected moods, instruments, and the binary feature vector
    print("Selected Moods:", active_moods)
    print("Active Instruments:", active_instruments)
    print("Binary Representation:", feature_vector)

    instrument_sequences = []
    instrument_names = []
    all_models_predictions = []
    initial_sequence = feature_vector.copy()
    generated_sequences = {instrument: [] for instrument in active_instruments}

    # Generate sequences using all models
    for instrument in active_instruments:
        if instrument in models:
            model = models[instrument]
            # Directly use the entire sequence generation function once per instrument
            generated_sequences[instrument] = generate_interdependent_sequence(
                {instrument: model}, initial_sequence, prefix_length, window_length, len(instrument_list), num_generate
            )

    # Process each instrument's generated sequences for MIDI conversion
    for instrument in active_instruments:
        if instrument in generated_sequences:
            token_sequence = generated_sequences[instrument]
            converted_sequence = [parse_token(token, reverse_vocab) for token in token_sequence]
            cleaned_sequence = clean_sequence(converted_sequence)
            instrument_sequences.append(cleaned_sequence)
            instrument_names.append(instrument)

    if instrument_sequences and instrument_names:
        create_midi_from_sequences(instrument_sequences, instrument_names, midi_file_path)
        midi_to_audio(midi_file_path, sound_font, output_audio_path)
        

# Submit button
submit_button = tk.Button(root, text="Generate", command=on_submit)
submit_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()