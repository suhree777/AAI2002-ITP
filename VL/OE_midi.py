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
            # If it's a single note, append its MIDI pitch
            pitches.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            # If it's a chord, append the MIDI pitches of all notes in the chord
            pitches.extend(n.midi for n in element.pitches)

    # Calculate the standard deviation of pitch occurrences to evaluate consistency
    pitch_variance = pd.Series(pitches).value_counts().std()
    return pitch_variance

def evaluate_temporal_structure(midi_stream):
    # Evaluate temporal structure by analyzing note durations and offsets
    durations = [n.duration.quarterLength for n in midi_stream.recurse().notes]
    offset_changes = [n.offset for n in midi_stream.recurse().notes]
    duration_consistency = pd.Series(durations).value_counts().std()
    offset_variance = pd.Series(offset_changes).diff().abs().mean()
    return duration_consistency, offset_variance

def evaluate_melodic_contour(midi_stream):
    # Evaluate melodic contour by the direction of melodic intervals
    notes = []
    for element in midi_stream.recurse().notes:
        if isinstance(element, note.Note):
            notes.append(element)
        elif isinstance(element, chord.Chord):
            # If it's a chord, take the top note (could also take the bass or any other strategy)
            notes.append(element.sortAscending().notes[-1])  # Taking the highest note

    # Calculate intervals between consecutive notes
    intervals = [interval.Interval(n1, n2).semitones for n1, n2 in zip(notes[:-1], notes[1:])]
    contour_changes = pd.Series(intervals).diff().abs().mean()
    return contour_changes

def evaluate_music(file_path):
    midi_stream = load_midi(file_path)
    pitch_consistency = evaluate_pitch_consistency(midi_stream)
    duration_consistency, offset_variance = evaluate_temporal_structure(midi_stream)
    melodic_contour = evaluate_melodic_contour(midi_stream)

    return {
        "File": os.path.basename(file_path),
        "Pitch Consistency": pitch_consistency,
        "Duration Consistency": duration_consistency,
        "Offset Variance": offset_variance,
        "Melodic Contour": melodic_contour
    }

def main():
    print("Current Working Directory:", os.getcwd())
    input_folder = 'VL/1_output'
    output_folder = 'VL/4_evaluation_results'
    results = []

    # Evaluate all MIDI files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            evaluation_result = evaluate_music(file_path)
            results.append(evaluation_result)

    results_df = pd.DataFrame(results)
    
    # Check if the results file already exists
    results_file_path = os.path.join(output_folder, 'midi_evaluation_results.csv')
    if os.path.exists(results_file_path):
        # Load existing results and append new results
        existing_df = pd.read_csv(results_file_path)
        combined_df = pd.concat([existing_df, results_df])
        # Remove duplicates based on the "File" column
        combined_df = combined_df.drop_duplicates(subset=["File"], keep='last')
        combined_df.to_csv(results_file_path, index=False)
        print(f"Results updated in {results_file_path}")
    else:
        # Save new results
        results_df.to_csv(results_file_path, index=False)
        print(f"Results saved in {results_file_path}")

if __name__ == "__main__":
    main()
