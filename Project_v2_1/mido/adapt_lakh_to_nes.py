import mido
import os
import pickle


def instrument_is_monophonic(track):
    notes = [msg for msg in track if msg.type in ['note_on', 'note_off']]
    notes.sort(key=lambda msg: msg.time)
    last_note_end = 0
    for msg in notes:
        if msg.type == 'note_on':
            if msg.time < last_note_end:
                return False
        elif msg.type == 'note_off':
            last_note_end = msg.time
    return True


def track_signature(track):
    return '-'.join([f"{msg.note}-{msg.velocity}" for msg in track if msg.type == 'note_on'])


def is_drum_track(track):
    for msg in track:
        if msg.type == 'program_change':
            return msg.channel == 9
    return False


def preprocess_midi_file(midi_file_path, output_dir):
    try:
        if os.path.getsize(midi_file_path) > (512 * 1024):
            print(f"Skipping large file: {midi_file_path}")
            return

        midi_file = mido.MidiFile(midi_file_path)

        midi_len = midi_file.length
        if midi_len < 5 or midi_len > 600:
            print(f"Skipping file with extreme length: {midi_file_path}")
            return

        nes_instruments = {
            'P1': {'min_pitch': 33, 'max_pitch': 108},
            'P2': {'min_pitch': 33, 'max_pitch': 108},
            'TR': {'min_pitch': 21, 'max_pitch': 108}
        }

        processed_tracks = []
        ensemble_length = 0
        track_signatures = set()
        for track in midi_file.tracks:
            if instrument_is_monophonic(track) and not is_drum_track(track):
                min_pitch = 128
                max_pitch = 0
                track_length = 0
                for msg in track:
                    if msg.type == 'note_on':
                        min_pitch = min(min_pitch, msg.note)
                        max_pitch = max(max_pitch, msg.note)
                        track_length = max(
                            track_length, msg.time + msg.velocity)
                for instrument, characteristics in nes_instruments.items():
                    if characteristics['min_pitch'] <= min_pitch and max_pitch <= characteristics['max_pitch']:
                        signature = track_signature(track)
                        if signature not in track_signatures:
                            track_signatures.add(signature)
                            processed_track = mido.MidiTrack()
                            processed_track.name = instrument
                            for msg in track:
                                if msg.type in ['note_on', 'note_off'] and msg.note >= characteristics['min_pitch'] and msg.note <= characteristics['max_pitch']:
                                    msg.time = max(0, msg.time)
                                    msg.time = int(
                                        msg.time * 44100 / 1000) * 1000 / 44100
                                    processed_track.append(msg)
                            processed_tracks.append(processed_track)
                            ensemble_length = max(
                                ensemble_length, track_length)
                            break

        max_length = 180
        if ensemble_length > max_length:
            for track in processed_tracks:
                for msg in track:
                    if msg.time > max_length:
                        track.remove(msg)

        new_midi = mido.MidiFile()
        new_midi.tracks.extend(processed_tracks)

        end_of_song = mido.MetaMessage(
            'end_of_track', time=int(max_length * 44100))
        new_midi.tracks.append(mido.MidiTrack([end_of_song]))

        output_file_path = os.path.join(
            output_dir, os.path.basename(midi_file_path))
        new_midi.save(output_file_path)
    except Exception as e:
        print(f"Error processing {midi_file_path}: {e}")


def preprocess_lmd_full(input_dir, output_dir, processed_folders_file):
    os.makedirs(output_dir, exist_ok=True)

    processed_folders = set()
    if os.path.exists(processed_folders_file):
        with open(processed_folders_file, 'rb') as f:
            processed_folders = pickle.load(f)

    sub_folders = [sub_folder for sub_folder in os.listdir(
        input_dir) if os.path.isdir(os.path.join(input_dir, sub_folder))]
    sub_folders = [sub_folder for sub_folder in sub_folders if sub_folder not in processed_folders]

    num_folders_to_process = 4
    sub_folders_to_process = sub_folders[:num_folders_to_process]

    for sub_folder in sub_folders_to_process:
        print(f"Processing folder: {sub_folder}")
        sub_folder_path = os.path.join(input_dir, sub_folder)
        for root, dirs, files in os.walk(sub_folder_path):
            for filename in files:
                if filename.endswith('.mid') or filename.endswith('.midi'):
                    midi_file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(root, input_dir)
                    file_output_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(file_output_dir, exist_ok=True)
                    preprocess_midi_file(midi_file_path, file_output_dir)
        processed_folders.add(sub_folder)
        with open(processed_folders_file, 'wb') as f:
            pickle.dump(processed_folders, f)


input_dir = 'lmd_full'
output_dir = 'mido/lmd_output'
processed_folders_file = 'mido/processed_folders.pkl'
preprocess_lmd_full(input_dir, output_dir, processed_folders_file)
