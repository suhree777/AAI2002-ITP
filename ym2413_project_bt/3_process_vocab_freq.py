import os
import json
import shutil
from collections import defaultdict

def load_json_files(dataset_path, instrument_vocab):
    event_vocab = defaultdict(int)  # Dictionary to count occurrences of each event type
    event_id = 2  # Start subsequent tokens from 1
    
    if os.path.isdir(dataset_path):
        print(f'Processing folder: {dataset_path}')
        for json_file in os.listdir(dataset_path):
            if json_file.endswith('.json'):
                json_path = os.path.join(dataset_path, json_file)
                with open(json_path, 'r') as file:
                    data = json.load(file)
                    instruments = data.get('instruments', {})
                    for instrument, events in instruments.items():
                        if instrument in instrument_vocab:  # Check if instrument is in the allowed list
                            for event in events:
                                event_key = tuple(sorted(event.items()))  # Create a hashable event key
                                if event_key not in event_vocab:
                                    event_vocab[event_key] = event_id
                                    event_id += 1  # Increment the ID for each new event
                                    #print(f'Adding new event to vocab: {event_key}')
    return event_vocab

def save_vocab(vocab, output_path):
    # Convert defaultdict to a regular dict for JSON serialization
    vocab_dict = {str(key): value for key, value in vocab.items()}
    with open(output_path, 'w') as file:
        json.dump(vocab_dict, file, indent=4)
    print(f'Vocabulary saved to {output_path}')

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
    return {tuple(eval(key)): value for key, value in vocab.items()}

def load_top_instruments(summary_path):
    with open(summary_path, 'r') as file:
        data = json.load(file)
        top_instruments = data['top_instrument_names']  # Accessing the specific key for instrument names
    return top_instruments

def transform_dataset(dataset_path, vocab, instrument_vocab, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    instrument_index_map = {instrument: index for index, instrument in enumerate(instrument_vocab)}
    num_instruments = len(instrument_vocab)
    processed_count = 0

    for json_file in os.listdir(dataset_path):
        if json_file.endswith('.json'):
            json_path = os.path.join(dataset_path, json_file)
            with open(json_path, 'r') as file:
                data = json.load(file)
                instruments = data.get('instruments', {})
                
                # Check if there are relevant instruments in this file
                relevant_instruments = {instr for instr in instruments if instr in instrument_index_map}
                if not relevant_instruments:
                    print(f"No relevant instruments found in {json_file}, skipping.")
                    continue  # Skip this file as it contains no relevant instruments

                transformed_instruments = {}
                instrument_vector = [0] * num_instruments
                for instrument, events in instruments.items():
                    if instrument in instrument_index_map:
                        instrument_index = instrument_index_map[instrument]
                        instrument_vector[instrument_index] = 1
                        transformed_events = [vocab[tuple(sorted(event.items()))] for event in events if tuple(sorted(event.items())) in vocab]
                        transformed_instruments[instrument] = transformed_events
                    
                transformed_data = {
                    'mood': data['mood'],
                    'instrument_vector': instrument_vector,
                    'instruments': transformed_instruments
                }

                output_file_path = os.path.join(output_folder, json_file)
                with open(output_file_path, 'w') as out_file:
                    json.dump(transformed_data, out_file, indent=4)
                processed_count += 1
    print(f'Total processed files: {processed_count}')

# Paths
dataset_path = 'ym2413_project_bt/2_feature_freq/data'
vocab_path = 'ym2413_project_bt/3_processed_freq/event_vocab.json'
processed_folder = 'ym2413_project_bt/3_processed_freq/data'
summary_path = 'ym2413_project_bt/3_processed_freq/summary.json'

"""dataset_path = 'ym2413_project_bt/2_feature_output_limited_freq880/data'
vocab_path = 'ym2413_project_bt/3_processed_feature_limited_freq880/event_vocab.json'
processed_folder = 'ym2413_project_bt/3_processed_feature_limited_freq880/data'
instrument_vocab_path = 'ym2413_project_bt/3_processed_feature_limited_freq880/instrument_list.json'"""

top_instruments  = load_top_instruments(summary_path)
event_vocab = load_json_files(dataset_path, top_instruments )
save_vocab(event_vocab, vocab_path)
vocab = load_vocab(vocab_path)
transform_dataset(dataset_path, vocab, top_instruments , processed_folder)