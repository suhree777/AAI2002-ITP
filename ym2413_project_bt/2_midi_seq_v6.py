import pretty_midi
import numpy as np
import os
import json
import re

instrument_vocab = set()

def round_features(features, precision=3):
    # Round all float values in the feature dictionaries to a specified precision
    rounded_features = []
    for feature in features:
        rounded_feature = {key: round(value, precision) if isinstance(value, float) else value
                           for key, value in feature.items()}
        rounded_features.append(rounded_feature)
    return rounded_features

def extract_midi_features(midi_file, instrument_vocab):
    print(f'Loading MIDI file: {midi_file}')
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    instrument_features = {}
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            instrument_vocab.add(instrument_name)  # Update the set with new instrument

            events = []
            current_event = []
            last_end_time = 0

            notes = sorted(instrument.notes, key=lambda x: x.start)
            for note in notes:
                # Record silence before the current note if there's a gap
                if note.start > last_end_time:
                    if last_end_time > 0:  # Avoid recording a silence at the start before any note
                        events.append({
                            'highest_pitch': 0,  # No pitch during silence
                            'highest_velocity': 0,  # No velocity during silence
                            'average_tempo': np.mean(np.interp([last_end_time, note.start],
                                                              midi_data.get_tempo_changes()[0],
                                                              midi_data.get_tempo_changes()[1])),
                            'duration': note.start - last_end_time
                        })
                    if current_event:
                        events.append(current_event)
                    current_event = []
                current_event.append(note)
                last_end_time = note.end
            # Handle the last current_event
            if current_event:
                events.append(current_event)

            notes_data = []
            for event in events:
                if isinstance(event, list):  # Check if event is a list of notes
                    start_time = min(note.start for note in event)
                    end_time = max(note.end for note in event)
                    highest_pitch = max(note.pitch for note in event)
                    highest_velocity = max(note.velocity for note in event)
                    tempos = np.interp([note.start for note in event],
                                       midi_data.get_tempo_changes()[0],
                                       midi_data.get_tempo_changes()[1])
                    event_features = {
                        'highest_pitch': highest_pitch,
                        'highest_velocity': highest_velocity,
                        'average_tempo': np.mean(tempos),
                        'duration': end_time - start_time
                    }
                    notes_data.append(event_features)
                else:
                    notes_data.append(event)  # Event is already a silence dictionary

            instrument_features[instrument_name] = round_features(notes_data)

    return instrument_features, instrument_vocab

def process_midi_dataset(dataset_path, output_folder, processed_folder):
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
                    # print(f"instrument_features print out{instrument_features}")
                    if not instrument_features or all(len(v) == 0 for v in instrument_features.values()):
                        print(f"No valid features found in {midi_file}, skipping.")
                        continue

                    track_id = os.path.splitext(midi_file)[0]
                    features = {
                        'track_id': track_id,
                        'mood': mood_label_clean,
                        'instruments': instrument_features
                    }
                    
                    output_subfolder = os.path.join(output_folder, mood_label)
                    os.makedirs(output_subfolder, exist_ok=True)
                    json_file_name = f'{track_id}.json'
                    output_file_path = os.path.join(output_subfolder, json_file_name)
                    
                    with open(output_file_path, 'w') as json_file:
                        json.dump(features, json_file, indent=4)
                    
                    print(f'Saved features to: {output_file_path}')

    # Save the instrument vocabulary to a JSON file
    instrument_dict = {instrument: idx for idx, instrument in enumerate(sorted(instrument_vocab))}
    with open(os.path.join(processed_folder, 'instrument_vocab.json'), 'w') as vocab_file:
        json.dump(instrument_dict, vocab_file, indent=4)

# Example usage setup
dataset_path = 'ym2413_project_bt/output'
output_json_path = 'ym2413_project_bt/feature_extracted'
processed_folder = 'ym2413_project_bt/processed_feature'
process_midi_dataset(dataset_path, output_json_path, processed_folder)
print('All features extracted and instrument vocabulary saved.')