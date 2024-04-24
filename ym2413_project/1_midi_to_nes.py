import csv
import os
import pretty_midi


pretty_midi.pretty_midi.MAX_TICK = 1e10

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


def preprocess_midi_file(midi_fp, output_dir, emotion_labels):
    try:
        # Skip large files to avoid processing overly complex MIDI files
        if os.path.getsize(midi_fp) > (512 * 1024):
            print(f"Skipping large file: {midi_fp}")
            return

        midi_data = pretty_midi.PrettyMIDI(midi_fp)

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
            output_dir, os.path.basename(midi_fp))
        new_midi.write(output_file_path)
    
    except Exception as e:
        print(f"Error processing {midi_fp}: {e}")


def preprocess_dataset(input_dir, output_dir):
    """Preprocess all MIDI files in the YM2413-MDB dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Read emotion annotation CSV file
    emotion_mapping = {}
    with open('music_dataset/YM2413-MDB-v1.0.2/emotion_annotation/verified_annotation.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            filename = row[0].replace('.wav', '.mid')  # Adjust filename extension
            emotions = row[2:]  # Taking emotion labels from "toptag_eng_verified" column
            emotion_mapping[filename] = emotions

    for filename in os.listdir(input_dir):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            midi_fp = os.path.join(input_dir, filename)

            emotion_labels = emotion_mapping.get(filename, [])  # Get emotion labels for the current MIDI file
            file_output_dir = os.path.join(output_dir, '_'.join(emotion_labels))  # Adjust output directory based on emotion labels
            os.makedirs(file_output_dir, exist_ok=True)
            preprocess_midi_file(midi_fp, file_output_dir, emotion_labels)


if __name__ == '__main__':
    input_dir = 'music_dataset/YM2413-MDB-v1.0.2/midi/adjust_tempo_remove_delayed_inst'
    output_dir = 'ym2413_project/output'
    preprocess_dataset(input_dir, output_dir)
