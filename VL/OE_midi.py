from music21 import converter, note, stream, chord, tempo
import os
import pandas as pd

def load_midi(file_path):
    # Load the MIDI file as a music21 stream
    midi_data = converter.parse(file_path)
    return midi_data

def evaluate_pitch_consistency(midi_stream):
    # Evaluate pitch consistency by analyzing note pitches
    pitches = [n.pitch.midi for n in midi_stream.recurse().notes]
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
    notes = [n for n in midi_stream.recurse().notes]
    intervals = [note.Interval(n1, n2).semitones for n1, n2 in zip(notes[:-1], notes[1:])]
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
    input_folder = '1_output'
    output_folder = '4_evaluation_results'
    results = []

    # Evaluate all MIDI files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            evaluation_result = evaluate_music(file_path)
            results.append(evaluation_result)
    
    # Save the results to a DataFrame and then to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'midi_evaluation_results.csv'), index=False)
    print(f"Results saved in {output_folder}/midi_evaluation_results.csv")

if __name__ == "__main__":
    main()
