import pretty_midi
import numpy as np
import os
import json
import re

def round_features(features, precision=4):
    # Round all float values in the feature dictionaries to a specified precision
    rounded_features = []
    for feature in features:
        rounded_features.append({
            'pitch': feature['pitch'],
            'velocity': feature['velocity'],
            'duration': round(feature['duration'], precision),
            'tempo': round(feature['tempo'], precision)
        })
    return rounded_features

def extract_midi_features(midi_file, instrument_vocab):
    print(f'Loading MIDI file: {midi_file}')
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    instrument_features = {}
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            print(f'Processing instrument: {instrument_name}')
            instrument_vocab.add(instrument_name)  # Update the set with new instrument
            
            notes_data = []
            for note in instrument.notes:
                note_features = {
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                }
                notes_data.append(note_features)
            
            # Compute tempos for notes
            tempo_changes = midi_data.get_tempo_changes()
            tempos = np.interp([note.start for note in instrument.notes], tempo_changes[0], tempo_changes[1])
            for note_data, tempo in zip(notes_data, tempos):
                note_data['tempo'] = tempo
            
            instrument_features[instrument_name] = round_features(notes_data)

    return instrument_features, instrument_vocab

def process_midi_dataset(dataset_path, output_folder, process_path):
    instrument_vocab = set()

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
                    instrument_features, instrument_vocab = extract_midi_features(midi_path, instrument_vocab)

                    if not instrument_features or all(len(v) == 0 for v in instrument_features.values()):
                        print(f"No valid features found in {midi_file}, skipping.")
                        continue

                    track_id = os.path.splitext(midi_file)[0]
                    features = {
                        'track_id': track_id,
                        'mood': mood_label_clean,
                        'instruments': instrument_features
                    }
                    
                    os.makedirs(output_folder, exist_ok=True)
                    json_file_name = f'{track_id}.json'
                    output_file_path = os.path.join(output_folder, json_file_name)
                    
                    with open(output_file_path, 'w') as json_file:
                        json.dump(features, json_file, indent=4)
                    
                    print(f'Saved features to: {output_file_path}')

    # Save the instrument vocabulary to a JSON file
    instrument_list = sorted(instrument_vocab)
    with open(os.path.join(process_path, 'instrument_list.json'), 'w') as vocab_file:
        json.dump(instrument_list, vocab_file, indent=4)

# Example usage setup
"""dataset_path = 'ym2413_project_bt/1_output'
output_json_path = 'ym2413_project_bt/2_feature_output/data'
process_path = 'ym2413_project_bt/3_processed_feature'"""

dataset_path = 'ym2413_project_bt/1_output_limited'
output_json_path = 'ym2413_project_bt/2_feature_output_limited/data'
process_path = 'ym2413_project_bt/3_processed_feature_limited'

process_midi_dataset(dataset_path, output_json_path, process_path)
print('All features extracted and instrument vocabulary saved.')

