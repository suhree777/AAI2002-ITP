import mido
import os

def encode_midi_to_events(midi_file_path):
    try:
        midi_file = mido.MidiFile(midi_file_path)
    except (EOFError, OSError) as e:
        print(f"Error reading {midi_file_path}: {e}")
        return None

    event_sequence = []

    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on':
                event_sequence.append(f'NOTE_ON_{msg.note}')
            elif msg.type == 'note_off':
                event_sequence.append(f'NOTE_OFF_{msg.note}')

    return event_sequence


def encode_directory_to_events(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        print(f"Processing folder: {os.path.basename(root)}")
        for filename in files:
            if filename.endswith('.mid') or filename.endswith('.midi'):
                midi_file_path = os.path.join(root, filename)
                event_sequence = encode_midi_to_events(midi_file_path)

                if event_sequence is not None:
                    # Save the event sequence to a file
                    relative_path = os.path.relpath(root, input_dir)
                    file_output_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(file_output_dir, exist_ok=True)
                    output_file_path = os.path.join(file_output_dir, f"{os.path.splitext(filename)[0]}.txt")
                    with open(output_file_path, 'w') as f:
                        f.write(' '.join(event_sequence))
                    # print(f"Encoded {filename}")

                    # Remove empty text files
                    if os.path.getsize(output_file_path) == 0:
                        os.remove(output_file_path)
                        print(f"Removed empty file: {output_file_path}")


input_dir = 'lmd_output'
output_dir = 'event_sequences'
encode_directory_to_events(input_dir, output_dir)
