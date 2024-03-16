import os
import pretty_midi


def extract_events(midi_file_path):
    """
    Extract musical events from a MIDI file for NES instrument channels and wait times.

    Args:
        midi_file_path (str): Path to the MIDI file.

    Returns:
        list: A list of musical event strings.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    events = []

    # Define channels for NES instruments and noise channel
    nes_channels = {
        'P1': [],
        'P2': [],
        'TR': [],
        'NO': []
    }

    # Extract notes for each instrument channel
    for instrument in midi_data.instruments:
        channel_name = instrument.name if instrument.name in nes_channels else 'NO'  # Default to noise channel if not specified
        for note in instrument.notes:
            nes_channels[channel_name].append((note.start, f"NOTEON_{note.pitch}", note.velocity))
            nes_channels[channel_name].append((note.end, f"NOTEOFF_{note.pitch}", note.velocity))

    # Sort events by time and add wait times
    for channel, notes in nes_channels.items():
        notes.sort(key=lambda x: x[0])
        last_time = 0
        for note in notes:
            wait_time = note[0] - last_time
            if wait_time > 0:
                events.append((note[0], f"WT_{int(wait_time * 100)}"))  # Convert wait time to integer for simplicity
            events.append((note[0], note[1]))
            last_time = note[0]

    events.sort(key=lambda x: x[0])  # Sort all events by time
    return [event[1] for event in events]  # Return only the event descriptions


def process_midi_dataset(midi_dataset_dir, max_files_per_subdir=200):
    """
    Process a limited number of MIDI files from each sub-folder in a dataset directory to extract musical events.

    Args:
        midi_dataset_dir (str): Path to the dataset directory containing MIDI files.
        max_files_per_subdir (int): Maximum number of MIDI files to process from each sub-folder.

    Returns:
        list: A list of encoded musical events.
    """
    all_events = []
    for root, dirs, files in os.walk(midi_dataset_dir):
        print(f"Processing folder: {root}")
        midi_files = [f for f in files if f.endswith(('.mid', '.midi'))][:max_files_per_subdir]  # Limit the number of files processed
        for midi_file in midi_files:
            midi_file_path = os.path.join(root, midi_file)
            events = extract_events(midi_file_path)
            all_events.extend(events)

    # Create a vocabulary to encode events
    unique_events = set(all_events)
    vocab = {event: idx for idx, event in enumerate(unique_events)}

    # Encode events using the vocabulary
    encoded_events = [vocab[event] for event in all_events]
    return encoded_events, vocab


def save_vocabulary(vocab, file_path):
    """
    Save the vocabulary to a text file.

    Args:
        vocab (dict): The vocabulary mapping musical events to integers.
        file_path (str): The path to the text file where the vocabulary will be saved.
    """
    with open(file_path, 'w') as f:
        for event, idx in vocab.items():
            f.write(f"{event}\t{idx}\n")


def save_encoded_events(encoded_events, file_path):
    """
    Save the encoded events to a text file.

    Args:
        encoded_events (list): The list of encoded musical events.
        file_path (str): The path to the text file where the encoded events will be saved.
    """
    with open(file_path, 'w') as f:
        for event in encoded_events:
            f.write(f"{event}\n")


if __name__ == '__main__':
    midi_dataset_dir = 'preprocess/lmd_output_batch'
    encoded_events, vocab = process_midi_dataset(midi_dataset_dir)
    print(f"Encoded events: {encoded_events[:10]}")  # Print the first 10 encoded events for demonstration
    print(f"Vocabulary size: {len(vocab)}")  # Print the size of the vocabulary

    vocab_file_path = 'preprocess/vocabulary.txt'
    save_vocabulary(vocab, vocab_file_path)
    print(f"Vocabulary saved to {vocab_file_path}")

    encoded_events_file_path = 'preprocess/encoded_events.txt'
    save_encoded_events(encoded_events, encoded_events_file_path)
    print(f"Encoded events saved to {encoded_events_file_path}")
