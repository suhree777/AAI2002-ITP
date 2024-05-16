import pretty_midi
import numpy as np
import os
import json
import re

def extract_midi_features(midi_file, target_rate=100):
    # Load MIDI file
    print(f'Loading MIDI file: {midi_file}')
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Get end time of the MIDI file
    end_time = midi_data.get_end_time()
    print(f'End time of MIDI file: {end_time:.2f} seconds')
    
    # Create a time array at the target rate
    time_array = np.arange(0, end_time, 1.0 / target_rate)
    
    # Initialize feature arrays
    pitch_features = []
    velocity_features = []
    duration_features = []
    tempo_features = []

    # Extract tempo changes
    tempos = midi_data.get_tempo_changes()
    tempo_times = tempos[0]
    tempo_values = tempos[1]
    
    for time in time_array:
        pitches = []
        velocities = []
        durations = []
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if note.start <= time < note.end:
                    pitches.append(note.pitch)
                    velocities.append(note.velocity)
                    durations.append(note.end - note.start)
        
        # Get the current tempo
        current_tempo = tempo_values[np.searchsorted(tempo_times, time) - 1]
        
        # Average features at each time point
        if pitches:
            pitch_features.append(np.mean(pitches))
            velocity_features.append(np.mean(velocities))
            duration_features.append(np.mean(durations))
        else:
            pitch_features.append(0)
            velocity_features.append(0)
            duration_features.append(0)
        
        tempo_features.append(current_tempo)
    
    # Combine features into a single array
    features = {
        'pitch': pitch_features,
        'velocity': velocity_features,
        'duration': duration_features,
        'tempo': tempo_features
    }
    
    print(f'Features extracted for {midi_file}')
    
    return features

def process_midi_dataset(dataset_path, output_folder, target_rate=20):
    for mood_folder in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood_folder)
        if os.path.isdir(mood_path):
            mood_label = os.path.basename(mood_path)
            # Remove prefix (e.g., "Q2_")
            mood_label_clean = re.sub(r'^Q\d+_', '', mood_label)
            print(f'Processing mood: {mood_label_clean}')
            for midi_file in os.listdir(mood_path):
                if midi_file.endswith('.mid'):
                    midi_path = os.path.join(mood_path, midi_file)
                    print(f'Processing file: {midi_path}')
                    features = extract_midi_features(midi_path, target_rate)
                    
                    # Add mood label to features
                    features['mood'] = mood_label_clean
                    
                    # Define output file path
                    json_file_name = f'{os.path.splitext(midi_file)[0]}.json'
                    output_file_path = os.path.join(output_folder, json_file_name)
                    
                    # Save features to JSON file
                    with open(output_file_path, 'w') as json_file:
                        json.dump(features, json_file, indent=4)
                    
                    print(f'Saved features to: {output_file_path}')

# Paths to your dataset and where to save features
dataset_path = 'ym2413_project_bt/output'
output_json_path = 'ym2413_project_bt/feature_output'

# Process the MIDI dataset and save to JSON
process_midi_dataset(dataset_path, output_json_path)
print('All features extracted')