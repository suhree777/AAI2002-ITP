import pretty_midi
import numpy as np
import os
import shutil
import json
import re
import matplotlib.pyplot as plt

def calculate_sample_rate(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    min_delta = float('inf')
    
    for instrument in midi_data.instruments:
        notes = sorted(instrument.notes, key=lambda note: note.start)
        for i in range(1, len(notes)):
            delta = notes[i].start - notes[i-1].start
            if delta > 0:
                min_delta = min(min_delta, delta)
    
    if min_delta == float('inf'):
        return 44100  # Default sample rate when no deltas are found
    
    frequency = 1 / min_delta
    return int(frequency * 2)  # Nyquist rate with a safety margin

def gather_sample_rates(dataset_path):
    sample_rates = {}
    for mood_folder in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood_folder)
        if os.path.isdir(mood_path):
            for midi_file in os.listdir(mood_path):
                if midi_file.endswith('.mid'):
                    midi_path = os.path.join(mood_path, midi_file)
                    rate = calculate_sample_rate(midi_path)
                    sample_rates[midi_path] = rate
    return sample_rates

def plot_sample_rates(rate_counter):
    rates = list(rate_counter.keys())
    counts = list(rate_counter.values())
    
    plt.figure(figsize=(12, 6))
    plt.scatter(rates, counts, color='blue')  # Use scatter plot for better visibility
    for i, txt in enumerate(counts):
        plt.annotate(txt, (rates[i], counts[i]), textcoords="offset points", xytext=(0,10), ha='center')  # Annotate each point
    
    plt.xlabel('Sample Rate (Hz)')
    plt.ylabel('Number of Files')
    plt.title('Distribution of Sample Rates Across MIDI Files')
    plt.grid(True)
    plt.show(block=False)  # Show non-blocking plot

def extract_midi_features(midi_file, target_rate, instrument_vocab, global_instrument_data):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    end_time = midi_data.get_end_time()
    print(f'End time of MIDI file: {end_time:.2f} seconds')
    
    # Create a time array at the target rate
    time_array = np.arange(0, end_time, 1.0 / target_rate)
    
    # Initialize dictionary to store features for each instrument
    instrument_features = {}
    instrument_counters = {}

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            base_instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            if base_instrument_name not in instrument_counters:
                instrument_counters[base_instrument_name] = 1
            else:
                instrument_counters[base_instrument_name] += 1
            
            if instrument_counters[base_instrument_name] > 1:
                instrument_name = f"{base_instrument_name}_{instrument_counters[base_instrument_name]}"
            else:
                instrument_name = base_instrument_name
            if instrument_name not in global_instrument_data:
                global_instrument_data[instrument_name] = {'count': 0, 'files': []}
            global_instrument_data[instrument_name]['count'] += 1
            global_instrument_data[instrument_name]['files'].append(midi_file)

            print(f'Processing instrument: {instrument_name}')
            instrument_vocab.add(instrument_name)

            instrument_data = []
            
            for time in time_array:
                notes_at_time = [
                    {'pitch': note.pitch,'velocity': note.velocity,}
                    for note in instrument.notes
                    if note.start <= time < note.end
                ]
                
                if not notes_at_time:
                    notes_at_time.append({'pitch': 0,'velocity': 0,})
                
                instrument_data.extend(notes_at_time)
            
            instrument_features[instrument_name] = instrument_data
    
    print(f'Features extracted for {midi_file}')
    return instrument_features, instrument_vocab, global_instrument_data

def process_midi_dataset(dataset_path, output_folder, process_path, target_rate, threshold, blacklist):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    instrument_vocab = set()
    global_instrument_data = {}
    processed_count = 0
    mood_labels = set()

    for mood_folder in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood_folder)
        if os.path.isdir(mood_path):
            mood_label = os.path.basename(mood_path)
            mood_label_clean = re.sub(r'^Q\d+_', '', mood_label)
            mood_labels.add(mood_label_clean)
            print(f'Processing mood: {mood_label_clean}')
            
            for midi_file in os.listdir(mood_path):
                if midi_file.endswith('.mid'):
                    midi_path = os.path.join(mood_path, midi_file)
                    if midi_path in blacklist:
                        continue  # Skip processing this file
                    processed_count += 1
                    print(f'Processing file: {midi_path}')
                    instrument_features, instrument_vocab, global_instrument_data  = extract_midi_features(midi_path, target_rate, instrument_vocab, global_instrument_data)
                    track_id = os.path.splitext(os.path.basename(midi_file))[0]
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
    # Identify the top 5 instruments
    top_instruments = sorted(global_instrument_data.items(), key=lambda x: x[1]['count'], reverse=True)[:threshold]
    top_instrument_names = [instr[0] for instr in top_instruments]  # Extract just the names

    return processed_count, list(mood_labels), top_instrument_names

def update_summary(summary_path, desired_rate, mood_labels, top_instrument_names):
    # Load the existing data from the summary file
    with open(summary_path, 'r') as file:
        data = json.load(file)

    # Update the data dictionary with new entries
    data['desired_sample_rate'] = desired_rate
    data['mood_labels'] = mood_labels
    data['top_instrument_names'] = top_instrument_names

    # Write the updated data back to the same file
    with open(summary_path, 'w') as file:
        json.dump(data, file, indent=4)
    
# Paths to your dataset and where to save features
dataset_path = 'ym2413_project_bt/1_output_freq'
output_json_path = 'ym2413_project_bt/2_feature_freq/data'
process_path = 'ym2413_project_bt/3_processed_freq'
summary_path = os.path.join(process_path, 'summary.json')

threshold = 4  # Instrument frequency limit for blacklisting

def main():
    sample_rates = gather_sample_rates(dataset_path)
    rate_counter = {}  # Dictionary to count occurrences of each sample rate
    for rate in sample_rates.values():
        if rate in rate_counter:
            rate_counter[rate] += 1
        else:
            rate_counter[rate] = 1
    print("Plotting unique sample rates and their counts:")
    plot_sample_rates(rate_counter)

    print("Unique Sample Rates Found and Their Counts:")
    for rate, count in sorted(rate_counter.items()):  # Sort by rate for readability
        print(f"{rate} Hz: {count} files")
    
    desired_rate = int(input("Enter your desired maximum sample rate: "))
    plt.close()

    blacklist = [path for path, rate in sample_rates.items() if rate > desired_rate]
    print("Files to be ignored based on the sample rate threshold:")
    for path in blacklist:
        print(path)

    # Process the MIDI dataset and save to JSON
    processed_count, mood_labels, top_instrument_names = process_midi_dataset(dataset_path, output_json_path, process_path, desired_rate, 4, blacklist)
    print(f'All features extracted. Total files processed: {processed_count}')

    update_summary(summary_path, desired_rate, mood_labels, top_instrument_names)
    print(f'Summary of the session saved to {summary_path}')

main()