import pretty_midi
import numpy as np
import os
import json
import re

instrument_vocab = set()

def round_features(features, precision=4):
    # Round all float values in the feature dictionaries to a specified precision
    rounded_features = []
    for feature in features:
        rounded_features.append({
            'pitch': feature['pitch'],
            'velocity': feature['velocity'],
            'duration': round(feature['duration'], precision),
            'tempo': round(feature.get('tempo', 0), precision)  # Use get to provide a default value
        })
    return rounded_features

def extract_midi_features(midi_file, instrument_vocab, precision=4):
    # print(f'Loading MIDI file: {midi_file}')
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    instrument_features = {}
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            # print(f'Processing instrument: {instrument_name}')
            instrument_vocab.add(instrument_name)  # Update the set with new instrument

            notes_data = []
            last_end_time = 0
            for note in instrument.notes:
                # Handle silence between notes
                if note.start > last_end_time:
                    silence_duration = note.start - last_end_time
                    silence_features = {
                        'pitch': 0,
                        'velocity': 0,
                        'duration': silence_duration,
                        'tempo': 0
                    }
                    notes_data.append(silence_features)

                note_features = {
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'duration': note.end - note.start,
                    'tempo': 0  # Initialize tempo with 0
                }
                notes_data.append(note_features)
                last_end_time = note.end

            # Compute tempos for notes
            tempo_changes = midi_data.get_tempo_changes()
            tempos = np.interp([note.start for note in instrument.notes], tempo_changes[0], tempo_changes[1])
            for note_data, tempo in zip([nd for nd in notes_data if nd['pitch'] != 0], tempos):
                note_data['tempo'] = tempo

            instrument_features[instrument_name] = round_features(notes_data, precision)
    return instrument_features, instrument_vocab

def process_midi_dataset(dataset_path, output_path, process_path, precision=4):
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
                    # print(f'Processing file: {midi_path}')
                    instrument_features, instrument_vocab = extract_midi_features(midi_path, instrument_vocab, precision)

                    if not instrument_features or all(len(v) == 0 for v in instrument_features.values()):
                        print(f"No valid features found in {midi_file}, skipping.")
                        continue

                    track_id = os.path.splitext(midi_file)[0]
                    features = {
                        'track_id': track_id,
                        'mood': mood_label_clean,
                        'instruments': instrument_features
                    }

                    output_subpath = os.path.join(output_path, mood_label_clean)
                    os.makedirs(output_subpath, exist_ok=True)
                    json_file_name = f'{track_id}.json'
                    output_file_path = os.path.join(output_subpath, json_file_name)

                    with open(output_file_path, 'w') as json_file:
                        json.dump(features, json_file, indent=4)

                    # print(f'Saved features to: {output_file_path}')

    os.makedirs(process_path, exist_ok=True)

    # Save the instrument vocabulary to a JSON file
    instrument_dict = {instrument: idx for idx, instrument in enumerate(sorted(instrument_vocab))}
    with open(os.path.join(process_path, 'instrument_vocab.json'), 'w') as vocab_file:
        json.dump(instrument_dict, vocab_file, indent=4)

if __name__ == '__main__':
    dataset_path = 'ym2413_project_xt/1_output'
    output_path = 'ym2413_project_xt/2_output_features'
    process_path = 'ym2413_project_xt/2.1_processed_features'
    
    process_midi_dataset(dataset_path, output_path, process_path)