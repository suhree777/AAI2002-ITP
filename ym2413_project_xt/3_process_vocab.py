import json
import os
from collections import defaultdict

def load_instrument_vocab(vocab_path):
    """Load the instrument vocabulary from a JSON file."""
    with open(vocab_path, 'r') as file:
        return json.load(file)

def load_instrument_specific_vocab(dataset_path):
    instrument_event_vocab = defaultdict(lambda: {"<PAD>": 0})  # Nested dictionary for each instrument
    instrument_event_id = defaultdict(lambda: 1)  # Separate ID counters for each instrument
    for mood in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood)
        if os.path.isdir(mood_path):
            print(f'Processing mood: {mood}')
            for json_file in os.listdir(mood_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(mood_path, json_file)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        instruments = data.get('instruments', {})
                        for instrument, events in instruments.items():
                            for event in events:
                                event_key = tuple(sorted(event.items()))  # Create a hashable event key
                                if event_key not in instrument_event_vocab[instrument]:
                                    instrument_event_vocab[instrument][event_key] = instrument_event_id[instrument]
                                    instrument_event_id[instrument] += 1
    return instrument_event_vocab

def save_vocab(vocab, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each instrument's vocabulary and save to separate files
    for instrument, events in vocab.items():
        vocab_dict = {str(key): value for key, value in events.items()}
        output_path = os.path.join(output_folder, f"{instrument}_vocab.json")
        with open(output_path, 'w') as file:
            json.dump(vocab_dict, file, indent=4)
        print(f'Vocabulary for {instrument} saved to {output_path}')

def load_all_vocabs(vocab_folder):
    """Load all vocabularies from the specified folder into a dictionary."""
    vocabs = {}
    for vocab_file in os.listdir(vocab_folder):
        if vocab_file.endswith('_vocab.json'):
            instrument = vocab_file.replace('_vocab.json', '')
            with open(os.path.join(vocab_folder, vocab_file), 'r') as file:
                vocabs[instrument] = json.load(file)

    # Convert keys from string to tuple, skip <PAD> token
    processed_vocabs = {}
    for instr, vocab in vocabs.items():
        vocab_dict = {}
        for key, value in vocab.items():
            if key != "<PAD>":  # Skip the PAD token
                try:
                    # Try to convert the string representation of the tuple to an actual tuple
                    tuple_key = tuple(eval(key))
                    vocab_dict[tuple_key] = value
                except SyntaxError:
                    # If there is a syntax error in eval, log or ignore
                    print(f"Error evaluating key {key}: Skipping")
        processed_vocabs[instr] = vocab_dict
    return processed_vocabs

def transform_dataset(dataset_path, vocabs, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for mood in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood)
        if os.path.isdir(mood_path):
            output_mood_path = os.path.join(output_folder, mood)
            os.makedirs(output_mood_path, exist_ok=True)
            for json_file in os.listdir(mood_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(mood_path, json_file)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        transformed_instruments = {}
                        for instrument, events in data['instruments'].items():
                            if instrument in vocabs:  # Check if there's a vocab for this instrument
                                vocab = vocabs[instrument]
                                transformed_events = [vocab.get(tuple(sorted(event.items())), -1) for event in events]  # Get event ID or -1 if not found
                                transformed_instruments[instrument] = transformed_events
                        transformed_data = {
                            'mood': data['mood'],
                            'instruments': transformed_instruments
                        }
                        output_file_path = os.path.join(output_mood_path, json_file)
                        with open(output_file_path, 'w') as out_file:
                            json.dump(transformed_data, out_file, indent=4)

if __name__ == '__main__':
    dataset_path = 'ym2413_project_xt/2_output_features'
    vocab_output_folder = 'ym2413_project_xt/3_processed_features/instrument_vocabs'
    processed_folder = 'ym2413_project_xt/3_processed_features/data'
    instrument_vocab_path = 'ym2413_project_xt/3_processed_features/instrument_vocab.json'

    # Load the data and build the vocabulary
    instrument_vocab = load_instrument_specific_vocab(dataset_path)
    save_vocab(instrument_vocab, vocab_output_folder)
    vocabs = load_all_vocabs(vocab_output_folder)
    
    # Transform the dataset using the loaded vocabularies
    transform_dataset(dataset_path, vocabs, processed_folder)
