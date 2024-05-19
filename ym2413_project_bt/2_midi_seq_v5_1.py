import pretty_midi
import numpy as np
import os
import json
import re

instrument_vocab = set()

def round_features(features, precision=7):
    # Round all float values in the feature dictionaries to a specified precision
    rounded_features = []
    for feature in features:
        rounded_features.append({
            'start_time': round(feature['start_time'], precision),
            'end_time': round(feature['end_time'], precision),
            'pitch': feature['pitch'],
            'velocity': feature['velocity'],
            'duration': round(feature['duration'], precision),
            'tempo': round(feature['tempo'], precision)
        })
    return rounded_features

def extract_midi_features(midi_file):
    print(f'Loading MIDI file: {midi_file}')
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    instrument_features = {}

    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # Ignore drum instruments
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            print(f'Processing instrument: {instrument_name}')
            instrument_vocab.add(instrument_name)  # Add to the global set
            
            notes_data = []
            for note in instrument.notes:
                note_features = {
                    'start_time': note.start,
                    'end_time': note.end,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                }
                notes_data.append(note_features)
            
            tempo_changes = midi_data.get_tempo_changes()
            tempos = np.interp([note.start for note in instrument.notes], tempo_changes[0], tempo_changes[1])

            for note_data, tempo in zip(notes_data, tempos):
                note_data['tempo'] = tempo
            
            instrument_features[instrument_name] = notes_data

    rounded_instrument_features = {inst: round_features(features) for inst, features in instrument_features.items()}
    print(f'Features extracted for {midi_file}')
    return rounded_instrument_features

def process_midi_dataset(dataset_path, output_folder):
    for mood_folder in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood_folder)
        if os.path.isdir(mood_path):
            mood_label = os.path.basename(mood_path)
            mood_label_clean = re.sub(r'^Q\d+_', '', mood_label)
            print(f'Processing mood: {mood_label_clean}')
            for midi_file in os.listdir(mood_path):
                if midi_file.endswith('.mid'):
                    midi_path = os.path.join(mood_path, midi_file)
                    print(f'Processing file: {midi_path}')
                    instrument_features = extract_midi_features(midi_path)
                    
                    # Skip saving if no instrument features were extracted
                    if not instrument_features or all(len(v) == 0 for v in instrument_features.values()):
                        print(f"No valid features found in {midi_file}, skipping.")
                        continue
                    
                    features = {
                        'mood': mood_label_clean,
                        'instruments': instrument_features
                    }
                    
                    output_subfolder = os.path.join(output_folder, mood_label)
                    os.makedirs(output_subfolder, exist_ok=True)
                    json_file_name = f'{os.path.splitext(midi_file)[0]}.json'
                    output_file_path = os.path.join(output_subfolder, json_file_name)
                    
                    with open(output_file_path, 'w') as json_file:
                        json.dump(features, json_file, indent=4)
                    
                    print(f'Saved features to: {output_file_path}')
    # Save the instrument vocabulary to a JSON file
    instrument_dict = {instrument: idx for idx, instrument in enumerate(sorted(instrument_vocab))}
    with open(os.path.join(output_folder, 'instrument_vocab.json'), 'w') as vocab_file:
        json.dump(instrument_dict, vocab_file, indent=4)


# Example usage setup
dataset_path = 'ym2413_project_bt/output'
output_json_path = 'ym2413_project_bt/feature_extracted'
process_midi_dataset(dataset_path, output_json_path)
print('All features extracted and instrument vocabulary saved.')
