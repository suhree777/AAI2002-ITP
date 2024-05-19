import os
import pretty_midi

def extract_events(midi_file_path, emotion_label):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    events = []
    for instrument in midi_data.instruments:
        channel_name = instrument.name if instrument.name in ['P1', 'P2', 'TR', 'NO'] else 'NO'
        for note in instrument.notes:
            events.append((f"NOTEON_{note.pitch}", note.velocity, note.start, emotion_label))
            events.append((f"NOTEOFF_{note.pitch}", note.velocity, note.end, emotion_label))

    events.sort(key=lambda x: x[2])  # Sort by time
    return events

def process_midi_dataset(midi_dataset_dir, max_files_per_emotion=75):
    all_events = []
    file_count = {}  # To track number of files processed per emotion

    for root, dirs, files in os.walk(midi_dataset_dir):
        emotion_label = os.path.basename(root)
        file_count[emotion_label] = file_count.get(emotion_label, 0)
        
        for midi_file in files:
            if midi_file.endswith(('.mid', '.midi')):
                if file_count[emotion_label] >= max_files_per_emotion:
                    continue  # Skip processing if the limit is reached
                midi_file_path = os.path.join(root, midi_file)
                events = extract_events(midi_file_path, emotion_label)
                all_events.extend(events)
                file_count[emotion_label] += 1  # Increment count for processed file

    event_emotion_tuples = [(event[0], event[3]) for event in all_events]
    unique_events = set(event_emotion_tuples)
    vocab = {event: idx for idx, event in enumerate(unique_events)}

    # Encode all events using the vocabulary
    encoded_events = [vocab[(event[0], event[3])] for event in all_events]
    return vocab, encoded_events

def save_vocabulary(vocab, file_path):
    with open(file_path, 'w') as f:
        for event, idx in vocab.items():
            f.write(f"('{event[0]}', '{event[1]}')\t{idx}\n")

def save_encoded_events(encoded_events, file_path):
    with open(file_path, 'w') as f:
        for event in encoded_events:
            f.write(f"{event}\n")

if __name__ == '__main__':
    midi_dataset_dir = 'ym2413_project/output'
    vocab, encoded_events = process_midi_dataset(midi_dataset_dir)

    vocab_file_path = 'ym2413_project/vocabulary.txt'
    save_vocabulary(vocab, vocab_file_path)
    print(f"Vocabulary saved to {vocab_file_path}")

    encoded_events_file_path = 'ym2413_project/encoded_events.txt'
    save_encoded_events(encoded_events, encoded_events_file_path)
    print(f"Encoded events saved to {encoded_events_file_path}")
