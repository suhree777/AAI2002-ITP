import pretty_midi
import os


def instrument_is_monophonic(instrument):
    """Check if a MIDI instrument is monophonic (no overlapping notes)."""
    notes = instrument.notes
    notes.sort(key=lambda note: note.start)
    last_note_end = 0
    for note in notes:
        if note.start < last_note_end:
            return False
        last_note_end = note.end
    return True


def track_signature(instrument):
    """Create a signature for a track based on its notes' pitches and velocities."""
    return '-'.join([f"{note.pitch}-{note.velocity}" for note in instrument.notes])


def preprocess_midi_file(midi_file_path, output_dir):
    try:
        # Skip large files to avoid processing overly complex MIDI files
        if os.path.getsize(midi_file_path) > (512 * 1024):
            print(f"Skipping large file: {midi_file_path}")
            return

        midi_data = pretty_midi.PrettyMIDI(midi_file_path)

        # Find the start time of the first note event
        first_note_start = min(note.start for instrument in midi_data.instruments for note in instrument.notes)

        # Shift all note events so that the first note starts at time 0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note.start = max(0, note.start - first_note_start)
                note.end = max(0, note.end - first_note_start)

        # Skip files with extreme lengths (too short or too long)
        midi_len = midi_data.get_end_time()
        if midi_len < 5 or midi_len > 600:
            print(f"Skipping file with extreme length: {midi_file_path}")
            return

        # Define NES instrument channels with their pitch ranges
        nes_instruments = {
            'P1': {'min_pitch': 33, 'max_pitch': 108},
            'P2': {'min_pitch': 33, 'max_pitch': 108},
            'TR': {'min_pitch': 21, 'max_pitch': 108},
            'NO': {'min_pitch': 0, 'max_pitch': 127}
        }

        processed_instruments = []
        ensemble_length = 0
        track_signatures = set()
        for instrument in midi_data.instruments:
            if instrument_is_monophonic(instrument) and not instrument.is_drum:
                min_pitch = min(note.pitch for note in instrument.notes)
                max_pitch = max(note.pitch for note in instrument.notes)
                track_length = max(note.end for note in instrument.notes)
                for name, characteristics in nes_instruments.items():
                    if characteristics['min_pitch'] <= min_pitch <= characteristics['max_pitch'] and \
                       characteristics['min_pitch'] <= max_pitch <= characteristics['max_pitch']:
                        signature = track_signature(instrument)
                        if signature not in track_signatures:
                            track_signatures.add(signature)
                            processed_instrument = pretty_midi.Instrument(
                                program=instrument.program, is_drum=instrument.is_drum, name=name)
                            for note in instrument.notes:
                                if characteristics['min_pitch'] <= note.pitch <= characteristics['max_pitch']:
                                    processed_instrument.notes.append(note)
                            processed_instruments.append(processed_instrument)
                            ensemble_length = max(
                                ensemble_length, track_length)
                            break

        # Limit the length of the ensemble to a maximum duration
        max_length = 180
        if ensemble_length > max_length:
            for instrument in processed_instruments:
                instrument.notes = [
                    note for note in instrument.notes if note.end <= max_length]

        # Create a new MIDI object and write the processed instruments to a new file
        new_midi = pretty_midi.PrettyMIDI()
        new_midi.instruments.extend(processed_instruments)
        output_file_path = os.path.join(
            output_dir, os.path.basename(midi_file_path))
        new_midi.write(output_file_path)
    except Exception as e:
        print(f"Error processing {midi_file_path}: {e}")


"""
Preprocess all MIDI files in all sub-folders of the `raw_data/lmd_full`
"""
# def preprocess_lmd_full(input_dir, output_dir):
#     """Preprocess all MIDI files in the raw dataset."""
#     os.makedirs(output_dir, exist_ok=True)

#     for root, dirs, files in os.walk(input_dir):
#         relative_path = os.path.relpath(root, input_dir)
        
#         # Skip the root directory itself
#         if relative_path == '.':
#             continue

#         print(f"Processing folder: {relative_path}")
#         for filename in files:
#             if filename.endswith('.mid') or filename.endswith('.midi'):
#                 midi_file_path = os.path.join(root, filename)
#                 file_output_dir = os.path.join(output_dir, relative_path)
#                 os.makedirs(file_output_dir, exist_ok=True)
#                 preprocess_midi_file(midi_file_path, file_output_dir)


"""
Batch preprocess all MIDI files in all sub-folders of the `raw_data/lmd_full`
"""
def preprocess_lmd_full(input_dir, output_dir, batch_size=3):
    """Preprocess all MIDI files in the raw dataset in batches."""
    os.makedirs(output_dir, exist_ok=True)

    # Load progress from checkpoint file
    checkpoint_file = os.path.join(output_dir, 'progress_checkpoint.txt')
    processed_folders = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_folders = set(f.read().splitlines())

    batch_count = 0
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)

        # Skip the root directory itself
        if relative_path == '.':
            continue

        if relative_path in processed_folders:
            continue  # Skip already processed folders

        print(f"Processing folder: {relative_path}")
        for filename in files:
            if filename.endswith('.mid') or filename.endswith('.midi'):
                midi_file_path = os.path.join(root, filename)
                file_output_dir = os.path.join(output_dir, relative_path)
                os.makedirs(file_output_dir, exist_ok=True)
                preprocess_midi_file(midi_file_path, file_output_dir)

        # Update progress
        processed_folders.add(relative_path)
        with open(checkpoint_file, 'a') as f:
            f.write(relative_path + '\n')

        batch_count += 1
        if batch_count >= batch_size:
            print("Batch limit reached. Stopping processing.")
            break


if __name__ == '__main__':
    # input_dir = 'rawdata/lmd_full'
    # output_dir = 'preprocess/lmd_output'
    input_dir = 'rawdata/lmd_full'
    output_dir = 'preprocess/lmd_output_batch'
    preprocess_lmd_full(input_dir, output_dir)
