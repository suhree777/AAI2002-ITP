import mido
import os
from random import choice

def preprocess_midi_file(midi_file_path, output_dir):
    try:
        midi_file = mido.MidiFile(midi_file_path)

        # Identify monophonic tracks and filter out-of-range instruments
        monophonic_tracks = []
        for track in midi_file.tracks:
            is_monophonic = True
            for msg in track:
                if msg.type == 'note_on':
                    if not (21 <= msg.note <= 108):
                        is_monophonic = False
                        break
            if is_monophonic:
                monophonic_tracks.append(track)

        # Randomly assign tracks to NES instruments
        nes_instruments = ['P1', 'P2', 'TR']
        assigned_tracks = {instr: [] for instr in nes_instruments}
        for track in monophonic_tracks:
            assigned_instrument = choice(nes_instruments)
            assigned_tracks[assigned_instrument].append(track)

        # Create a new MIDI file with the processed tracks
        new_midi = mido.MidiFile()
        for instrument, tracks in assigned_tracks.items():
            for track in tracks:
                new_midi.tracks.append(track)

        # Save the processed MIDI file
        output_file_path = os.path.join(output_dir, os.path.basename(midi_file_path))
        new_midi.save(output_file_path)
    except Exception as e:
        print(f"Error processing {midi_file_path}: {e}")


def preprocess_lmd_full(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Recursively traverse the input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.mid') or filename.endswith('.midi'):
                midi_file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_dir)
                file_output_dir = os.path.join(output_dir, relative_path)
                os.makedirs(file_output_dir, exist_ok=True)
                preprocess_midi_file(midi_file_path, file_output_dir)
                print(f"Processed {midi_file_path}")


input_dir = 'lmd_full'
output_dir = 'lmd_output'
preprocess_lmd_full(input_dir, output_dir)
