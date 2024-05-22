import pretty_midi
import numpy as np
import os
import json
import re

def round_value(value, precision=2):
    return round(value, precision)

def round_features(features, precision=2):
    # Round all float values in the feature dictionaries to a specified precision
    rounded_features = []
    for feature in features:
        rounded_feature = {key: round(value, precision) if isinstance(value, float) else value
                           for key, value in feature.items()}
        rounded_features.append(rounded_feature)
    return rounded_features

def extract_midi_features(midi_file, pitch_vocab, velocity_vocab, instrument_vocab):
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
                if note.start > last_end_time:
                    if last_end_time > 0:
                        silence_duration = note.start - last_end_time
                        rounded_silence_duration = round_value(silence_duration)
                        events.append({
                            'highest_pitch': 0,
                            'highest_velocity': 0,
                            'duration': rounded_silence_duration
                        })
                        pitch_vocab.add((rounded_silence_duration, 0))
                        velocity_vocab.add((rounded_silence_duration, 0))
                    if current_event:
                        events.append(current_event)
                    current_event = []
                current_event.append(note)
                last_end_time = note.end
            if current_event:
                events.append(current_event)

            notes_data = []
            for event in events:
                if isinstance(event, list):
                    start_time = min(note.start for note in event)
                    end_time = max(note.end for note in event)
                    event_duration = end_time - start_time
                    rounded_event_duration = round_value(event_duration)
                    highest_pitch = max(note.pitch for note in event)
                    highest_velocity = max(note.velocity for note in event)
                    event_features = {
                        'highest_pitch': highest_pitch,
                        'highest_velocity': highest_velocity,
                        'duration': rounded_event_duration
                    }
                    notes_data.append(event_features)
                    pitch_vocab.add((rounded_event_duration, highest_pitch))
                    velocity_vocab.add((rounded_event_duration, highest_velocity))
                else:
                    notes_data.append(event)

            instrument_features[instrument_name] = round_features(notes_data)

    return instrument_features, pitch_vocab, velocity_vocab, instrument_vocab

def save_vocabularies(pitch_vocab, velocity_vocab, instrument_vocab, processed_folder):
    pitch_dict = {f"duration: {dur}, pitch: {pitch}": idx for idx, (dur, pitch) in enumerate(sorted(pitch_vocab))}
    velocity_dict = {f"duration: {dur}, velocity: {vel}": idx for idx, (dur, vel) in enumerate(sorted(velocity_vocab))}
    instrument_dict = {instrument: idx for idx, instrument in enumerate(sorted(instrument_vocab))}

    with open(os.path.join(processed_folder, 'pitch_vocab.json'), 'w') as pitch_file:
        json.dump(pitch_dict, pitch_file, indent=4)
    with open(os.path.join(processed_folder, 'velocity_vocab.json'), 'w') as velocity_file:
        json.dump(velocity_dict, velocity_file, indent=4)
    with open(os.path.join(processed_folder, 'instrument_vocab.json'), 'w') as vocab_file:
        json.dump(instrument_dict, vocab_file, indent=4)
    print('Pitch, velocity, and instrument vocabularies saved.')

def process_midi_dataset(dataset_path, output_folder, processed_folder):
    # Initialize vocabularies
    pitch_vocab = set()
    velocity_vocab = set()
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
                    instrument_features, pitch_vocab, velocity_vocab, instrument_vocab = extract_midi_features(midi_path, pitch_vocab, velocity_vocab, instrument_vocab)
                    
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
    save_vocabularies(pitch_vocab, velocity_vocab, instrument_vocab, processed_folder)
    print('All features extracted and vocabularies saved.')

# Example usage setup
dataset_path = 'ym2413_project_bt/output'
output_json_path = 'ym2413_project_bt/feature_extracted'
processed_folder = 'ym2413_project_bt/processed_feature'
process_midi_dataset(dataset_path, output_json_path, processed_folder)
print('All features extracted and instrument vocabulary saved.')