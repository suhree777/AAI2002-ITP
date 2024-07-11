from music21 import converter, note, chord, interval
import os
import pandas as pd

def load_midi(file_path):
    # Load the MIDI file as a music21 stream
    midi_data = converter.parse(file_path)
    return midi_data

def evaluate_pitch_consistency(midi_stream):
    # Initialize an empty list to collect pitches
    pitches = []

    # Loop through all notes and chords in the MIDI stream
    for element in midi_stream.recurse():
        if isinstance(element, note.Note):
            pitches.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            pitches.extend(n.midi for n in element.pitches)

    pitch_variance = pd.Series(pitches).value_counts().std()
    return pitch_variance

def evaluate_temporal_structure(midi_stream):
    durations = [n.duration.quarterLength for n in midi_stream.recurse().notes]
    offset_changes = [n.offset for n in midi_stream.recurse().notes]
    duration_consistency = pd.Series(durations).value_counts().std()
    offset_variance = pd.Series(offset_changes).diff().abs().mean()
    return duration_consistency, offset_variance

def evaluate_melodic_contour(midi_stream):
    notes = []
    for element in midi_stream.recurse().notes:
        if isinstance(element, note.Note):
            notes.append(element)
        elif isinstance(element, chord.Chord):
            notes.append(element.sortAscending().notes[-1])

    intervals = [interval.Interval(n1, n2).semitones for n1, n2 in zip(notes[:-1], notes[1:])]
    contour_changes = pd.Series(intervals).diff().abs().mean()
    return contour_changes

def evaluate_note_density_and_dynamic_range(midi_stream):
    notes = [n for n in midi_stream.recurse().notes]
    total_duration = sum(n.duration.quarterLength for n in notes)
    note_density = len(notes) / total_duration if total_duration else 0

    velocities = [n.volume.velocity for n in notes]
    dynamic_range = max(velocities) - min(velocities) if velocities else 0
    return note_density, dynamic_range

def evaluate_music(file_path):
    midi_stream = load_midi(file_path)
    pitch_consistency = evaluate_pitch_consistency(midi_stream)
    duration_consistency, offset_variance = evaluate_temporal_structure(midi_stream)
    melodic_contour = evaluate_melodic_contour(midi_stream)
    note_density, dynamic_range = evaluate_note_density_and_dynamic_range(midi_stream)

    return {
        "File": os.path.basename(file_path),
        "Pitch Consistency": pitch_consistency,
        "Duration Consistency": duration_consistency,
        "Offset Variance": offset_variance,
        "Melodic Contour": melodic_contour,
        "Note Density": note_density,
        "Dynamic Range": dynamic_range
    }

def main():
    print("Current Working Directory:", os.getcwd())
    input_folder = 'VL/1_output'
    output_folder = 'VL/4_evaluation_results'
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            evaluation_result = evaluate_music(file_path)
            results.append(evaluation_result)

    results_df = pd.DataFrame(results)
    
    results_file_path = os.path.join(output_folder, 'midi_evaluation_results.csv')
    if os.path.exists(results_file_path):
        existing_df = pd.read_csv(results_file_path)
        combined_df = pd.concat([existing_df, results_df])
        combined_df = combined_df.drop_duplicates(subset=["File"], keep='last')
        combined_df.to_csv(results_file_path, index=False)
        print(f"Results updated in {results_file_path}")
    else:
        results_df.to_csv(results_file_path, index=False)
        print(f"Results saved in {results_file_path}")

if __name__ == "__main__":
    main()
