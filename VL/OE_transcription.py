from music21 import converter, environment, note, tempo
import mido
import os

# Set up the environment for music21
env = environment.UserSettings()
env['musicxmlPath'] = r'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
env['musescoreDirectPNGPath'] = r'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

def load_midi_with_mido(file_path):
    """Load a MIDI file using the mido library."""
    return mido.MidiFile(file_path)

def parse_midi(file_path):
    """Load and parse a MIDI file into a music21 stream."""
    return converter.parse(file_path)

def analyze_midi_file(midi_file):
    """Prints out MIDI messages from the file."""
    print(f"Analyzing MIDI file: {midi_file.filename}")
    for i, track in enumerate(midi_file.tracks):
        print(f"Track {i}: {track.name}")
        for msg in track:
            print(msg)

def generate_music_script(midi_file):
    """Generate a simple script showing note sequences from a MIDI file."""
    notes = []
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note = mido.get_note_name(msg.note)
                notes.append(f"{note} played at velocity {msg.velocity}")
    return notes

def main():
    input_folder = 'VL/1_output'
    for filename in os.listdir(input_folder):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            midi_data = load_midi_with_mido(file_path)
            analyze_midi_file(midi_data)
            notes_script = generate_music_script(midi_data)
            print("\nGenerated Music Script:")
            for note in notes_script:
                print(note)

if __name__ == "__main__":
    main()
