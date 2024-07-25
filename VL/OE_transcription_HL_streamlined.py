import os
import pandas as pd
import mido

# Function to load a MIDI file
def load_midi(file_path):
    try:
        return mido.MidiFile(file_path)
    except Exception as e:
        print(f"Error loading MIDI file {file_path}: {e}")
        return None

# Function to evaluate pitch consistency
def evaluate_pitch_consistency(pitches):
    if not pitches:
        return 0
    pitch_variance = pd.Series(pitches).value_counts().std()
    return pitch_variance

# Function to evaluate temporal structure
def evaluate_temporal_structure(durations, offsets):
    if not durations or not offsets:
        return 0, 0
    duration_consistency = pd.Series(durations).value_counts().std()
    offset_variance = pd.Series(offsets).diff().abs().mean()
    return duration_consistency, offset_variance

# Function to evaluate melodic contour
def evaluate_melodic_contour(intervals):
    if not intervals:
        return 0
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
    normalized_value = (value - min_value) / (max_value - min_value)
    return max(0, min(normalized_value, 1))

# Function to classify mood based on the evaluated features using Russell's Emotion Circumplex model
def classify_mood(pitch_consistency, duration_consistency, offset_variance, melodic_contour, note_density, dynamic_range):
    norm_pitch_consistency = normalize_feature(pitch_consistency, 0, 20)
    norm_duration_consistency = normalize_feature(duration_consistency, 0, 30)
    norm_offset_variance = normalize_feature(offset_variance, 0, 200)
    norm_melodic_contour = normalize_feature(melodic_contour, 0, 20)
    norm_note_density = normalize_feature(note_density, 1, 10)
    norm_dynamic_range = normalize_feature(dynamic_range, 10, 100)
    
    arousal = (norm_pitch_consistency + norm_duration_consistency + norm_offset_variance + norm_note_density + norm_dynamic_range) / 5
    valence = (1 - norm_pitch_consistency + 1 - norm_duration_consistency + 1 - norm_offset_variance + norm_note_density + norm_dynamic_range) / 5

    if arousal > 0.5 and valence > 0.5:
        return "Happy"
    elif arousal > 0.5 and valence <= 0.5:
        return "Tensional"
    elif arousal <= 0.5 and valence > 0.5:
        return "Peaceful"
    elif arousal <= 0.5 and valence <= 0.5:
        return "Sad"
    else:
        return "Neutral"

# Define thresholds and weights for evaluation
THRESHOLDS = {
    "Melody Score": (0, 1),
    "Harmony Score": (0, 100),
    "Rhythm Score": (0, 1),
    "Overall Structure Score": (0, 100)
}

WEIGHTS = {
    "Melody Score": 2,
    "Harmony Score": 1,
    "Rhythm Score": 1,
    "Overall Structure Score": 2
}

# Function to evaluate a single row of metrics for high-level quality score
def evaluate_row_high_level(row):
    score = 0
    max_score = 0
    for metric in ["Melody Score", "Harmony Score", "Rhythm Score", "Overall Structure Score"]:
        value = row[metric]
        if value is not None:
            normalized_value = normalize_feature(value, THRESHOLDS[metric][0], THRESHOLDS[metric][1])
            score += WEIGHTS[metric] * normalized_value
            max_score += WEIGHTS[metric]
    return (score / max_score) * 100 if max_score != 0 else 0

# Function to evaluate the dataframe using high-level quality score
def evaluate_music_df_high_level(df):
    df['Quality Score'] = df.apply(evaluate_row_high_level, axis=1)
    return df

# Function to dynamically evaluate music based on selected features
def evaluate_music(file_path, selected_features):
    midi_file = load_midi(file_path)
    if midi_file is None:
        return None

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

    pitch_consistency = evaluate_pitch_consistency(pitches)
    duration_consistency, offset_variance = evaluate_temporal_structure(durations, offsets)
    melodic_contour = evaluate_melodic_contour(intervals)
    note_density, dynamic_range = evaluate_note_density_and_dynamic_range(notes, velocities, total_duration)

    # Calculate high-level features
    melody_score = evaluate_melody(melodic_contour, pitch_consistency)
    harmony_score = evaluate_harmony(chords, len(midi_file.tracks))
    rhythm_score = evaluate_rhythm(duration_consistency, offset_variance)
    structure_score = evaluate_overall_structure(note_density, dynamic_range)

    # Collect feature values based on selected features
    feature_values = {
        "Melody Score": melody_score,
        "Harmony Score": harmony_score,
        "Rhythm Score": rhythm_score,
        "Overall Structure Score": structure_score
    }

    # Classify mood based on the evaluated features (using all features needed for mood classification)
    predicted_mood = classify_mood(pitch_consistency, duration_consistency, offset_variance, melodic_contour, note_density, dynamic_range)

    result = {
        "File": os.path.basename(file_path),
        "Predicted Mood": predicted_mood,
        "Melody Score": feature_values["Melody Score"],
        "Harmony Score": feature_values["Harmony Score"],
        "Rhythm Score": feature_values["Rhythm Score"],
        "Overall Structure Score": feature_values["Overall Structure Score"]
    }

    return result

def main():
    print("Current Working Directory:", os.getcwd())
    input_folder = 'VL/1_output/samples'
    output_folder = 'VL/4_evaluation_results'
    selected_features = [
        'melody', 'harmony', 'rhythm', 'overall_structure'
    ]
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            evaluation_result = evaluate_music(file_path, selected_features)
            if evaluation_result:
                results.append(evaluation_result)

    results_df = pd.DataFrame(results)

    # Removing unnecessary columns
    results_df = results_df[["File", "Predicted Mood", "Melody Score", "Harmony Score", "Rhythm Score", "Overall Structure Score"]]

    # Evaluate the quality of the music (high-level)
    results_df = evaluate_music_df_high_level(results_df)

    results_file_path = os.path.join(output_folder, 'midi_evaluation_results_hl_streamlined.csv')
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
