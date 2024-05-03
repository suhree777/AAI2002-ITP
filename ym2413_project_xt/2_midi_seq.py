import os
import pretty_midi

def extract_events(midi_file_path, emotion_label):
    """
    Extracts musical events from a MIDI file, optimized to handle event sequencing efficiently and consistently.
    Args:
        midi_file_path (str): Path to the MIDI file.
        emotion_label (str): Associated emotion label from the MIDI file's directory.
    Returns:
        list: A list of musical events with their respective timing and emotion label.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    events = []
    last_end_time = 0

    # Extract events maintaining the order they appear in the MIDI
    for instrument in midi_data.instruments:
        for note in sorted(instrument.notes, key=lambda x: x.start):  # Sort by start time for consistency
            wait_time = int(note.start - last_end_time)
            if wait_time > 0:
                events.append(f"WT_{wait_time}")
            events.append(f"{instrument.name}_NOTEON_{note.pitch}")
            events.append(f"{instrument.name}_NOTEOFF_{note.pitch}")
            last_end_time = note.end

    return events

def process_midi_dataset(midi_dataset_dir, max_files_per_emotion=75):
    """
    Processes MIDI files by extracting events and limiting processing to a maximum number per emotion.
    Args:
        midi_dataset_dir (str): Directory containing subdirectories for each emotion, each with MIDI files.
        max_files_per_emotion (int): Maximum number of MIDI files to process per emotion.
    Returns:
        dict: A dictionary of vocabularies for each emotion label.
        dict: A dictionary of encoded events for each emotion label.
    """
    all_events = {}
    vocab_by_emotion = {}
    encoded_events_by_emotion = {}

    for root, dirs, files in os.walk(midi_dataset_dir):
        emotion_label = os.path.basename(root)
        all_events[emotion_label] = []
        file_count = 0
        
        for midi_file in files:
            if midi_file.endswith(('.mid', '.midi')) and file_count < max_files_per_emotion:
                midi_file_path = os.path.join(root, midi_file)
                events = extract_events(midi_file_path, emotion_label)
                all_events[emotion_label].extend(events)
                file_count += 1  # Increment count for processed file

        # Create vocabulary and encode events
        unique_events = sorted(set(all_events[emotion_label]))
        vocab = {event: idx for idx, event in enumerate(unique_events)}
        vocab_by_emotion[emotion_label] = vocab
        encoded_events = [vocab[event] for event in all_events[emotion_label]]
        encoded_events_by_emotion[emotion_label] = encoded_events

    return vocab_by_emotion, encoded_events_by_emotion

def save_vocabulary(vocab_by_emotion, output_dir):
    """
    Saves the vocabulary of events for each emotion to separate text files.
    Args:
        vocab_by_emotion (dict): Dictionaries containing event vocabularies categorized by emotion.
        output_dir (str): Output directory for saving vocabulary files.
    """
    for emotion_label, vocab in vocab_by_emotion.items():
        file_path = os.path.join(output_dir, f'vocabulary_{emotion_label}.txt')
        with open(file_path, 'w') as f:
            sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[0]))  # Sort by instrument
            for idx, event in enumerate(sorted_vocab.keys()):
                f.write(f"{event}\t{idx}\n")

def save_encoded_events(encoded_events_by_emotion, output_dir):
    """
    Saves the encoded events for each emotion to separate text files.
    Args:
        encoded_events_by_emotion (dict): Dictionaries containing encoded events categorized by emotion.
        output_dir (str): Output directory for saving encoded event files.
    """
    for emotion_label, encoded_events in encoded_events_by_emotion.items():
        file_path = os.path.join(output_dir, f'encoded_events_{emotion_label}.txt')
        with open(file_path, 'w') as f:
            for index in encoded_events:
                f.write(f"{index}\n")

if __name__ == '__main__':
    midi_dataset_dir = 'ym2413_project/output'
    output_dir = 'ym2413_project/emotion_data'

    vocab_by_emotion, encoded_events_by_emotion = process_midi_dataset(midi_dataset_dir)

    os.makedirs(output_dir, exist_ok=True)

    save_vocabulary(vocab_by_emotion, output_dir)
    print("Vocabulary saved for each emotion.")

    save_encoded_events(encoded_events_by_emotion, output_dir)
    print("Encoded events saved for each emotion.")
