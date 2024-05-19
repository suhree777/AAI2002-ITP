import os
import pretty_midi
import numpy as np
import json
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_events(midi_file_path, emotion):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        features = []

        # Extract tempo and adjust times by tempo
        tempo_changes = midi_data.get_tempo_changes()
        last_tempo = midi_data.estimate_tempo() if len(tempo_changes[1]) == 0 else tempo_changes[1][0]

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                duration = end_time - start_time
                pitch = note.pitch
                velocity = note.velocity

                # Add basic note information
                features.append({
                    'emotion': emotion,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'pitch': pitch,
                    'velocity': velocity,
                    'instrument': instrument.program,
                    'is_drum': instrument.is_drum,
                    'tempo': last_tempo
                })

        return features
    except Exception as e:
        logging.error(f"Error processing {midi_file_path}: {str(e)}")
        return []

def process_midi_dataset(midi_dataset_dir):
    all_features = []
    for root, _, files in os.walk(midi_dataset_dir):
        emotion = os.path.basename(root)  # Extracts the emotion from the folder name
        for file in files:
            if file.endswith(('.mid', '.midi')):
                file_path = os.path.join(root, file)
                logging.info(f"Processing file: {file_path}")
                file_features = extract_events(file_path, emotion)
                all_features.extend(file_features)
    
    for feature in all_features:
        unique_pitches.add(feature['pitch'])
        unique_velocity.add(feature['velocity'])
        unique_instruments.add(feature['instrument'])
        unique_durations.add(feature['duration'])  # Consider binning durations if too varied

    return unique_pitches, unique_velocity, unique_instruments, unique_durations

def process_individual_midi(file_path, vocab_pitches, vocab_instruments, vocab_durations):
    features = extract_events(file_path, "emotion")  # Ensure 'extract_events' handles MIDI extraction per your need
    encoded_features = encode_features(features, vocab_pitches, vocab_instruments, vocab_durations)
    return encoded_features
    
def process_all_midis(directory_path, vocab_pitches, vocab_instruments, vocab_durations):
    all_encoded_features = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_path = os.path.join(root, file)
                encoded_features = process_individual_midi(file_path, vocab_pitches, vocab_instruments, vocab_durations)
                all_encoded_features.append(encoded_features)
    return all_encoded_features

def encode_features(features, vocab_pitches, vocab_instruments, vocab_durations):
    encoded_features = []
    for feature in features:
        encoded_feature = {
            'pitch': vocab_pitches[str(feature['pitch'])],
            'instrument': vocab_instruments[str(feature['instrument'])],
            'duration': vocab_durations[str(feature['duration'])]
        }
        encoded_features.append(encoded_feature)
    return encoded_features

def save_vocabulary(vocab, file_path):
    vocab = {}
    next_index = 0
    for item in vocab:
        # Convert numpy integers to Python integers
        if isinstance(item, np.integer):
            item = int(item)
        vocab[item] = next_index
        next_index += 1
    with open(file_path, 'w') as f:
        json.dump(vocab, f, indent=4)

def create_vocabulary(unique_set):
    vocab = {}
    next_index = 0
    for item in unique_set:
        # Convert numpy integers to Python integers
        if isinstance(item, np.integer):
            item = int(item)
        vocab[item] = next_index
        next_index += 1
    return vocab

if __name__ == '__main__':
    midi_dataset_dir = 'ym2413_project_bt/output'
    unique_pitches, unique_velocity, unique_instruments, unique_durations = process_midi_dataset(midi_dataset_dir)

    # Save vocabularies to separate JSON files
    save_vocabulary(unique_pitches, 'ym2413_project_bt/vocab_pitches.json')
    save_vocabulary(unique_velocity, 'ym2413_project_bt/vocab_pitches.json')
    save_vocabulary(unique_instruments, 'ym2413_project_bt/vocab_instruments.json')
    save_vocabulary(unique_durations, 'ym2413_project_bt/vocab_durations.json')

    print("Vocabularies saved.")
