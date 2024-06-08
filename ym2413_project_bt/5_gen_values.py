import os
import json
import ast
import pretty_midi
import numpy as np
from tensorflow.keras.models import load_model
from midi2audio import FluidSynth

midi_file = pretty_midi.PrettyMIDI()

model = load_model('ym2413_project_bt/my_lstm_model_full_095.keras')

def generate_sequence(model, seed_sequence, length_of_generation):
    generated_sequence = list(seed_sequence)  # Copy to avoid altering the original seed
    current_sequence = seed_sequence.copy()
    
    for _ in range(length_of_generation):
        # Predict the next token from the current sequence
        prediction_probs = model.predict(np.array([current_sequence]))[0]
        predicted_token = np.argmax(prediction_probs)  # Choose the most likely next token
        
        # Append the predicted token to the generated sequence
        generated_sequence.append(predicted_token)
        
        # Update the sequence to include the new token
        current_sequence = generated_sequence[-len(seed_sequence):]  # Keep it the same length as the seed sequence

    return generated_sequence

seed_sequence = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
generated_sequence = generate_sequence(model, seed_sequence, 150)
print(f"generated_sequence is as follows {generated_sequence}")

instrument_name = 'Acoustic Bass'
vocab_file_path = os.path.join('ym2413_project_bt/3_processed_feature_limited/instrument_vocabs/', f"{instrument_name}_vocab.json")

with open(vocab_file_path, 'r') as f:
    vocab = json.load(f)

# Create a reverse vocabulary
reverse_vocab = {v: k for k, v in vocab.items()}

# Convert the generated sequence to values
def parse_token(token):
    token_str = reverse_vocab.get(token, "<PAD>")
    try:
        if token_str == "<PAD>":
            return "PAD"
        else:
            return ast.literal_eval(token_str)  # Safely evaluate string to tuple
    except SyntaxError:
        return "Invalid Token"  # Handle any tokens that might still cause issues

converted_sequence = [parse_token(token) for token in generated_sequence[20:]]

# Printing converted values
for token, value in zip(generated_sequence[20:], converted_sequence):
    print(f"Token: {token}, Value: {value}")

# Convert structured feature sets into simple tuples
simple_features = []
for feature_set in converted_sequence:
    # Create a dictionary from the tuple of tuples
    feature_dict = dict(feature_set)
    
    # Extract values directly from the dictionary
    duration = feature_dict['duration']
    pitch = feature_dict['pitch']
    tempo = feature_dict['tempo']
    velocity = feature_dict['velocity']

    # Append as a simple tuple
    simple_feature = (duration, pitch, tempo, velocity)
    simple_features.append(simple_feature)

# Display the simplified feature list
print(simple_features)

# Create a PrettyMIDI object
midi_file = pretty_midi.PrettyMIDI()

# Create an instrument instance (e.g., Acoustic Bass)
instrument_program = pretty_midi.instrument_name_to_program('Acoustic Bass')
instrument = pretty_midi.Instrument(program=instrument_program)

# Add notes to the instrument
current_time = 0.0
for duration, pitch, tempo, velocity in simple_features:
    # Create a note
    note = pretty_midi.Note(
        velocity=int(velocity),
        pitch=int(pitch),
        start=current_time,
        end=current_time + duration
    )
    instrument.notes.append(note)
    current_time += duration  # Move to the next note start time

# Add the instrument to the PrettyMIDI object
midi_file.instruments.append(instrument)

# Save the MIDI file
midi_file.write('ym2413_project_bt/output.mid')

def midi_to_audio(midi_file_path, sound_font, output_audio_path):
    # Create a synthesizer object with the sound font
    fs = FluidSynth(sound_font)
    fs.midi_to_audio(midi_file_path, output_audio_path)
    print(f'Converted audio saved to {output_audio_path}')

# Specify paths and sound font
midi_file_path = 'ym2413_project_bt/output.mid'  # Path to your MIDI file
sound_font = 'ym2413_project_bt/chiptune_soundfont_4.0.sf2'  # Path to your sound font file
output_audio_path = 'ym2413_project_bt/output.wav'  # Path to save the output audio file

# Convert MIDI to audio
midi_to_audio(midi_file_path, sound_font, output_audio_path)