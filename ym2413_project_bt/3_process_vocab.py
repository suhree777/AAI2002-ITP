import os
import json
from collections import defaultdict

def load_json_files(dataset_path):
    event_vocab = defaultdict(int)  # Dictionary to count occurrences of each event type
    event_id = 0
    
    for mood_folder in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood_folder)
        if os.path.isdir(mood_path):
            print(f'Processing folder: {mood_folder}')
            for json_file in os.listdir(mood_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(mood_path, json_file)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        instruments = data.get('instruments', {})
                        for instrument, events in instruments.items():
                            for event in events:
                                event_key = tuple(sorted(event.items()))  # Create a hashable event key
                                if event_key not in event_vocab:
                                    event_vocab[event_key] = event_id
                                    event_id += 1
                                    #print(f'Adding new event to vocab: {event_key}')
    return event_vocab

def save_vocab(vocab, output_path):
    # Convert defaultdict to a regular dict for JSON serialization
    vocab_dict = {str(key): value for key, value in vocab.items()}
    with open(output_path, 'w') as file:
        json.dump(vocab_dict, file, indent=4)
    print(f'Vocabulary saved to {output_path}')

# Path to the dataset directory
dataset_path = 'ym2413_project_bt/feature_extracted'
vocab_path = 'ym2413_project_bt/processed_feature/event_vocab.json'

# Load the data and build the vocabulary
event_vocab = load_json_files(dataset_path)

# Save the vocabulary to a JSON file
save_vocab(event_vocab, vocab_path)

def load_vocab(vocab_path):
    """Load the vocabulary from a JSON file."""
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
    # Convert string keys back to tuples if needed
    return {tuple(eval(key)): value for key, value in vocab.items()}


def transform_dataset(dataset_path, vocab, output_folder):
    """Transform the dataset using the loaded vocabulary."""
    for mood_folder in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood_folder)
        if os.path.isdir(mood_path):
            output_subfolder = os.path.join(output_folder, mood_folder)
            os.makedirs(output_subfolder, exist_ok=True)
            for json_file in os.listdir(mood_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(mood_path, json_file)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        transformed_instruments = {}
                        for instrument, events in data['instruments'].items():
                            transformed_events = [vocab[tuple(sorted(event.items()))] for event in events if tuple(sorted(event.items())) in vocab]
                            transformed_instruments[instrument] = transformed_events
                        transformed_data = {
                            'mood': data['mood'],
                            'instruments': transformed_instruments
                        }
                        output_file_path = os.path.join(output_subfolder, json_file)
                        with open(output_file_path, 'w') as out_file:
                            json.dump(transformed_data, out_file, indent=4)
                    # print(f'Saved transformed data to: {output_file_path}')

# Paths
processed_folder = 'ym2413_project_bt/processed_feature'
vocab = load_vocab(vocab_path)
transform_dataset(dataset_path, vocab, processed_folder)