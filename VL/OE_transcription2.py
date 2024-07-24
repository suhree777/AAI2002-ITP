import os
import pandas as pd
import mido

# Function to load a MIDI file
def load_midi(file_path):
    return mido.MidiFile(file_path)

# Function to evaluate pitch consistency
def evaluate_pitch_consistency(pitches):
    pitch_variance = pd.Series(pitches).value_counts().std()
    return pitch_variance

# Function to evaluate temporal structure
def evaluate_temporal_structure(durations, offsets):
    duration_consistency = pd.Series(durations).value_counts().std()
    offset_variance = pd.Series(offsets).diff().abs().mean()
    return duration_consistency, offset_variance

# Function to evaluate melodic contour
def evaluate_melodic_contour(intervals):
    contour_changes = pd.Series(intervals).diff().abs().mean()
    return contour_changes

# Function to evaluate note density and dynamic range
def evaluate_note_density_and_dynamic_range(notes, velocities, total_duration):
    note_density = len(notes) / total_duration if total_duration else 0
    dynamic_range = max(velocities) - min(velocities) if velocities else 0
    return note_density, dynamic_range

# Function to evaluate melody
def evaluate_melody(melodic_contour, pitch_consistency):
    melody_score = (1 / melodic_contour if melodic_contour != 0 else 0) + (1 / pitch_consistency if pitch_consistency != 0 else 0)
    return melody_score

# Function to evaluate harmony
def evaluate_harmony(chords, total_tracks):
    harmony_score = chords / total_tracks if total_tracks != 0 else 0
    return harmony_score

# Function to evaluate rhythm
def evaluate_rhythm(duration_consistency, offset_variance):
    rhythm_score = (1 / duration_consistency if duration_consistency != 0 else 0) + (1 / offset_variance if offset_variance != 0 else 0)
    return rhythm_score

# Function to evaluate overall structure
def evaluate_overall_structure(note_density, dynamic_range):
    structure_score = note_density + dynamic_range
    return structure_score

# Function to normalize feature values
def normalize_feature(value, min_value, max_value):
    if max_value == min_value:
        return 0
    return (value - min_value) / (max_value - min_value)

# Function to classify mood based on the evaluated features using Russell's Emotion Circumplex model
def classify_mood(pitch_consistency, duration_consistency, offset_variance, melodic_contour, note_density, dynamic_range):
    # Normalize features
    norm_pitch_consistency = normalize_feature(pitch_consistency, 0, 20)
    norm_duration_consistency = normalize_feature(duration_consistency, 0, 30)
    norm_offset_variance = normalize_feature(offset_variance, 0, 200)
    norm_melodic_contour = normalize_feature(melodic_contour, 0, 20)
    norm_note_density = normalize_feature(note_density, 1, 10)
    norm_dynamic_range = normalize_feature(dynamic_range, 10, 100)
    
    # Calculate arousal and valence scores
    arousal = (norm_pitch_consistency + norm_duration_consistency + norm_offset_variance + norm_note_density + norm_dynamic_range) / 5
    valence = (1 - norm_pitch_consistency + 1 - norm_duration_consistency + 1 - norm_offset_variance + norm_note_density + norm_dynamic_range) / 5

    # Determine mood based on arousal and valence
    if arousal > 0.5 and valence > 0.5:
        return "Happy"
    elif arousal > 0.5 and valence <= 0.5:
        return "Angry"
    elif arousal <= 0.5 and valence > 0.5:
        return "Relaxed"
    elif arousal <= 0.5 and valence <= 0.5:
        return "Sad"
    else:
        return "Neutral"

# Define thresholds and weights for evaluation
THRESHOLDS = {
    "Pitch Consistency": (0, 20),
    "Duration Consistency": (0, 30),
    "Offset Variance": (0, 200),
    "Melodic Contour": (0, 20),
    "Note Density": (1, 10),
    "Dynamic Range": (10, 100),
    "Melody Score": (0, 1),
    "Harmony Score": (0, 100),
    "Rhythm Score": (0, 1),
    "Overall Structure Score": (0, 100)
}

WEIGHTS = {
    "Pitch Consistency": 1,
    "Duration Consistency": 1,
    "Offset Variance": 1,
    "Melodic Contour": 1,
    "Note Density": 1,
    "Dynamic Range": 1,
    "Melody Score": 2,
    "Harmony Score": 1,
    "Rhythm Score": 1,
    "Overall Structure Score": 2
}

# Function to evaluate a single row of metrics
def evaluate_row(row):
    score = 0
    for metric, (low, high) in THRESHOLDS.items():
        value = row[metric]
        if low <= value <= high:
            score += WEIGHTS[metric] * (1 - abs(value - (low + high) / 2) / ((high - low) / 2))
        else:
            score -= WEIGHTS[metric]
    return score

# Function to evaluate the dataframe
def evaluate_music_df(df):
    df['Quality Score'] = df.apply(evaluate_row, axis=1)
    return df

# Function to dynamically evaluate music based on selected features
def evaluate_music(file_path, selected_features):
    midi_file = load_midi(file_path)

    pitches = []
    durations = []
    offsets = []
    notes = []
    velocities = []
    intervals = []
    time = 0
    chords = 0

    for track in midi_file.tracks:
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.append(msg.note)
                durations.append(msg.time)
                offsets.append(time)
                notes.append(msg)
                velocities.append(msg.velocity)
                if len(track) > 1 and track[1].type == 'note_on':
                    chords += 1

    intervals = [pitches[i] - pitches[i-1] for i in range(1, len(pitches))]
    total_duration = midi_file.length

    feature_values = []

    if 'pitch_consistency' in selected_features:
        pitch_consistency = evaluate_pitch_consistency(pitches)
        feature_values.append(pitch_consistency)

    if 'temporal_structure' in selected_features:
        duration_consistency, offset_variance = evaluate_temporal_structure(durations, offsets)
        feature_values.extend([duration_consistency, offset_variance])

    if 'melodic_contour' in selected_features:
        melodic_contour = evaluate_melodic_contour(intervals)
        feature_values.append(melodic_contour)

    if 'note_density_dynamic_range' in selected_features:
        note_density, dynamic_range = evaluate_note_density_and_dynamic_range(notes, velocities, total_duration)
        feature_values.extend([note_density, dynamic_range])

    if 'melody' in selected_features:
        melodic_contour = evaluate_melodic_contour(intervals)
        pitch_consistency = evaluate_pitch_consistency(pitches)
        melody_score = evaluate_melody(melodic_contour, pitch_consistency)
        feature_values.append(melody_score)

    if 'harmony' in selected_features:
        harmony_score = evaluate_harmony(chords, len(midi_file.tracks))
        feature_values.append(harmony_score)

    if 'rhythm' in selected_features:
        duration_consistency, offset_variance = evaluate_temporal_structure(durations, offsets)
        rhythm_score = evaluate_rhythm(duration_consistency, offset_variance)
        feature_values.append(rhythm_score)

    if 'overall_structure' in selected_features:
        note_density, dynamic_range = evaluate_note_density_and_dynamic_range(notes, velocities, total_duration)
        structure_score = evaluate_overall_structure(note_density, dynamic_range)
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
    # input_folder = 'VL/1_output'
    # input_folder = 'VL/1_output/samples'
    input_folder = 'VL/1_output/gen samples'
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

    # Evaluate the quality of the music
    results_df = evaluate_music_df(results_df)

    # results_file_path = os.path.join(output_folder, 'midi_evaluation_results21.csv')
    # results_file_path = os.path.join(output_folder, 'midi_evaluation_results22.csv')
    results_file_path = os.path.join(output_folder, 'midi_evaluation_results23.csv')
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
