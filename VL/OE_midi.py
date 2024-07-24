import os
import pandas as pd
import mido

# Function to load a MIDI file
def load_midi(file_path):
    return mido.MidiFile(file_path)

# Function to evaluate pitch consistency
def evaluate_pitch_consistency(midi_file):
    pitches = []
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.append(msg.note)
    pitch_variance = pd.Series(pitches).value_counts().std()
    return pitch_variance

# Function to evaluate temporal structure
def evaluate_temporal_structure(midi_file):
    durations = []
    offsets = []
    time = 0
    for track in midi_file.tracks:
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                durations.append(msg.time)
                offsets.append(time)
    duration_consistency = pd.Series(durations).value_counts().std()
    offset_variance = pd.Series(offsets).diff().abs().mean()
    return duration_consistency, offset_variance

# Function to evaluate melodic contour
def evaluate_melodic_contour(midi_file):
    notes = []
    time = 0
    for track in midi_file.tracks:
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
    intervals = [notes[i] - notes[i-1] for i in range(1, len(notes))]
    contour_changes = pd.Series(intervals).diff().abs().mean()
    return contour_changes

# Function to evaluate note density and dynamic range
def evaluate_note_density_and_dynamic_range(midi_file):
    notes = []
    velocities = []
    total_duration = midi_file.length
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg)
                velocities.append(msg.velocity)
    note_density = len(notes) / total_duration if total_duration else 0
    dynamic_range = max(velocities) - min(velocities) if velocities else 0
    return note_density, dynamic_range

# Function to evaluate melody
def evaluate_melody(midi_file):
    # Assessing the smoothness, hierarchy, and adherence to music theory fundamentals
    # Placeholder: Using melodic contour and pitch consistency as proxies
    melodic_contour = evaluate_melodic_contour(midi_file)
    pitch_consistency = evaluate_pitch_consistency(midi_file)
    melody_score = (1 / melodic_contour if melodic_contour != 0 else 0) + (1 / pitch_consistency if pitch_consistency != 0 else 0)
    return melody_score

# Function to evaluate harmony
def evaluate_harmony(midi_file):
    # Evaluating the coordination and harmony
    # Placeholder: Counting simultaneous notes (chords) as a simple harmony measure
    chords = 0
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if len(track) > 1 and track[1].type == 'note_on':
                    chords += 1
    harmony_score = chords / len(midi_file.tracks)
    return harmony_score

# Function to evaluate rhythm
def evaluate_rhythm(midi_file):
    # Examining the accuracy and regularity of rhythm
    # Placeholder: Using duration consistency and offset variance
    duration_consistency, offset_variance = evaluate_temporal_structure(midi_file)
    rhythm_score = (1 / duration_consistency if duration_consistency != 0 else 0) + (1 / offset_variance if offset_variance != 0 else 0)
    return rhythm_score

# Function to evaluate overall structure
def evaluate_overall_structure(midi_file):
    # Analyzing the tightness, logic, and innovation
    # Placeholder: Using note density and dynamic range
    note_density, dynamic_range = evaluate_note_density_and_dynamic_range(midi_file)
    structure_score = note_density + dynamic_range
    return structure_score

# Function to classify mood based on the evaluated features
def classify_mood(pitch_consistency, duration_consistency, offset_variance, melodic_contour, note_density, dynamic_range):
    if pitch_consistency < 10 and duration_consistency < 0.5 and offset_variance < 0.1:
        return "Relaxed"
    elif pitch_consistency > 20 and duration_consistency > 0.7 and offset_variance > 0.2:
        return "Angry"
    elif note_density < 5 and dynamic_range < 30:
        return "Sad"
    elif note_density > 10 and dynamic_range > 40:
        return "Happy"
    else:
        return "Neutral"

# Function to dynamically evaluate music based on selected features
def evaluate_music(file_path, selected_features):
    midi_file = load_midi(file_path)
    feature_values = []

    if 'pitch_consistency' in selected_features:
        pitch_consistency = evaluate_pitch_consistency(midi_file)
        feature_values.append(pitch_consistency)

    if 'temporal_structure' in selected_features:
        duration_consistency, offset_variance = evaluate_temporal_structure(midi_file)
        feature_values.extend([duration_consistency, offset_variance])

    if 'melodic_contour' in selected_features:
        melodic_contour = evaluate_melodic_contour(midi_file)
        feature_values.append(melodic_contour)

    if 'note_density_dynamic_range' in selected_features:
        note_density, dynamic_range = evaluate_note_density_and_dynamic_range(midi_file)
        feature_values.extend([note_density, dynamic_range])

    if 'melody' in selected_features:
        melody_score = evaluate_melody(midi_file)
        feature_values.append(melody_score)

    if 'harmony' in selected_features:
        harmony_score = evaluate_harmony(midi_file)
        feature_values.append(harmony_score)

    if 'rhythm' in selected_features:
        rhythm_score = evaluate_rhythm(midi_file)
        feature_values.append(rhythm_score)

    if 'overall_structure' in selected_features:
        structure_score = evaluate_overall_structure(midi_file)
        feature_values.append(structure_score)

    predicted_mood = classify_mood(*feature_values[:6])  # Assuming first 6 features are used for mood classification

    result = {
        "File": os.path.basename(file_path),
        "Predicted Mood": predicted_mood,
        "Pitch Consistency": feature_values[0] if 'pitch_consistency' in selected_features else None,
        "Duration Consistency": feature_values[1] if 'temporal_structure' in selected_features else None,
        "Offset Variance": feature_values[2] if 'temporal_structure' in selected_features else None,
        "Melodic Contour": feature_values[3] if 'melodic_contour' in selected_features else None,
        "Note Density": feature_values[4] if 'note_density_dynamic_range' in selected_features else None,
        "Dynamic Range": feature_values[5] if 'note_density_dynamic_range' in selected_features else None,
        "Melody Score": feature_values[6] if 'melody' in selected_features else None,
        "Harmony Score": feature_values[7] if 'harmony' in selected_features else None,
        "Rhythm Score": feature_values[8] if 'rhythm' in selected_features else None,
        "Overall Structure Score": feature_values[9] if 'overall_structure' in selected_features else None,
    }

    return result

def main():
    print("Current Working Directory:", os.getcwd())
    input_folder = 'VL/1_output'
    output_folder = 'VL/4_evaluation_results'
    selected_features = [
        'pitch_consistency', 'temporal_structure', 'melodic_contour', 'note_density_dynamic_range',
        'melody', 'harmony', 'rhythm', 'overall_structure'
    ]
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            evaluation_result = evaluate_music(file_path, selected_features)
            results.append(evaluation_result)

    results_df = pd.DataFrame(results)

    # Removing unnecessary columns
    results_df.drop(columns=['pitch_consistency', 'temporal_structure', 'melodic_contour', 'note_density_dynamic_range'], inplace=True, errors='ignore')

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
