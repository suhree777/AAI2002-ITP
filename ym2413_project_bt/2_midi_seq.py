import os
import pretty_midi
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def extract_events(midi_file_path, emotion_label):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    events = []
    last_end_time = {pitch: 0 for pitch in range(128)}  # Track end times for all pitches

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = note.start
            duration = note.end - note.start
            time_since_last_note = start_time - last_end_time[note.pitch]
            last_end_time[note.pitch] = note.end

            events.append((f"NOTEON_{note.pitch}", note.velocity, duration, time_since_last_note, emotion_label))

    events.sort(key=lambda x: x[2])  # Sort by time
    return events

def process_midi_dataset(midi_dataset_dir, max_files_per_emotion=75):
    all_events = []
    file_count = {}  # To track number of files processed per emotion
    scaler = MinMaxScaler()

    for root, dirs, files in os.walk(midi_dataset_dir):
        emotion_label = os.path.basename(root)
        file_count[emotion_label] = file_count.get(emotion_label, 0)

        for midi_file in files:
            if midi_file.endswith(('.mid', '.midi')):
                if file_count[emotion_label] >= max_files_per_emotion:
                    continue  # Skip processing if the limit is reached
                midi_file_path = os.path.join(root, midi_file)
                print(f"Processing file: {midi_file_path}")
                events = extract_events(midi_file_path, emotion_label)
                all_events.extend(events)
                file_count[emotion_label] += 1  # Increment count for processed file

    # Normalize feature columns (assuming columns 1 to 3 are features to be normalized)
    feature_data = np.array([event[1:4] for event in all_events])
    normalized_features = scaler.fit_transform(feature_data)

    # Reconstruct events with normalized features
    normalized_events = [(all_events[i][0], *normalized_features[i], all_events[i][-1]) for i in range(len(all_events))]

    event_emotion_tuples = [(event[0], event[4]) for event in normalized_events]
    unique_events = set(event_emotion_tuples)
    vocab = {event: idx for idx, event in enumerate(unique_events)}

    # Encode all events using the vocabulary
    encoded_events = [vocab[(event[0], event[4])] for event in normalized_events]
    return vocab, encoded_events


def save_vocabulary(vocab, file_path):
    with open(file_path, 'w') as f:
        for event, idx in sorted(vocab.items(), key=lambda item: item[1]):  # Ensure consistent order
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
