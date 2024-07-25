import os
import pandas as pd
import mido
import numpy as np

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
def normalize_feature(value, mean, std):
    if std == 0:
        return 0
    normalized_value = (value - mean) / std
    # Ensure normalized values are within the range [0, 1]
    return max(0, min(normalized_value, 1))

# Function to classify mood based on the evaluated features using Russell's Emotion Circumplex model
def classify_mood(pitch_consistency, duration_consistency, offset_variance, melodic_contour, note_density, dynamic_range,
                  means, stds):
    norm_pitch_consistency = normalize_feature(pitch_consistency, means['Pitch Consistency'], stds['Pitch Consistency'])
    norm_duration_consistency = normalize_feature(duration_consistency, means['Duration Consistency'], stds['Duration Consistency'])
    norm_offset_variance = normalize_feature(offset_variance, means['Offset Variance'], stds['Offset Variance'])
    norm_melodic_contour = normalize_feature(melodic_contour, means['Melodic Contour'], stds['Melodic Contour'])
    norm_note_density = normalize_feature(note_density, means['Note Density'], stds['Note Density'])
    norm_dynamic_range = normalize_feature(dynamic_range, means['Dynamic Range'], stds['Dynamic Range'])
    
    arousal = (norm_pitch_consistency + norm_duration_consistency + norm_offset_variance + norm_note_density + norm_dynamic_range) / 5
    valence = (1 - norm_pitch_consistency + 1 - norm_duration_consistency + 1 - norm_offset_variance + norm_note_density + norm_dynamic_range) / 5

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

# Function to impute missing values with the mean
def impute_missing_values(df, metric):
    mean_value = df[metric].mean()
    df[metric] = df[metric].fillna(mean_value)
    return df

# Function to calculate mean and std for normalization
def calculate_mean_std(df, metric):
    return df[metric].mean(), df[metric].std()

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
def evaluate_row_high_level(row, means, stds):
    score = 0
    max_score = 0
    for metric, weight in WEIGHTS.items():
        value = row[metric]
        if value is not None:
            normalized_value = normalize_feature(value, means[metric], stds[metric])
            print(f"{metric}: Value={value}, Mean={means[metric]}, Std={stds[metric]}, Normalized={normalized_value}")
            score += weight * normalized_value
            max_score += weight
    print(f"Score={score}, Max Score={max_score}")
    return (score / max_score) * 100 if max_score != 0 else 0

# Function to evaluate the dataframe using high-level quality score
def evaluate_music_df_high_level(df, means, stds):
    df['Quality Score'] = df.apply(evaluate_row_high_level, axis=1, means=means, stds=stds)
    return df

# Function to dynamically evaluate music based on selected features
def evaluate_music(file_path):
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
        "Pitch Consistency": pitch_consistency,
        "Duration Consistency": duration_consistency,
        "Offset Variance": offset_variance,
        "Melodic Contour": melodic_contour,
        "Note Density": note_density,
        "Dynamic Range": dynamic_range,
        "Melody Score": melody_score,
        "Harmony Score": harmony_score,
        "Rhythm Score": rhythm_score,
        "Overall Structure Score": structure_score
    }

    return feature_values

def main():
    print("Current Working Directory:", os.getcwd())
    input_folder = 'VL/1_output/samples'
    output_folder = 'VL/4_evaluation_results'
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(input_folder, filename)
            feature_values = evaluate_music(file_path)
            if feature_values:
                results.append({
                    "File": os.path.basename(file_path),
                    **feature_values
                })

    results_df = pd.DataFrame(results)

    # Impute missing values with the mean
    for metric in ["Pitch Consistency", "Duration Consistency", "Offset Variance", "Melodic Contour", "Note Density", "Dynamic Range",
                   "Melody Score", "Harmony Score", "Rhythm Score", "Overall Structure Score"]:
        results_df = impute_missing_values(results_df, metric)

    # Calculate mean and std for normalization
    means = {}
    stds = {}
    for metric in ["Pitch Consistency", "Duration Consistency", "Offset Variance", "Melodic Contour", "Note Density", "Dynamic Range",
                   "Melody Score", "Harmony Score", "Rhythm Score", "Overall Structure Score"]:
        means[metric], stds[metric] = calculate_mean_std(results_df, metric)

    # Classify mood and calculate quality score
    classified_results = []
    for index, row in results_df.iterrows():
        pitch_consistency = row["Pitch Consistency"]
        duration_consistency = row["Duration Consistency"]
        offset_variance = row["Offset Variance"]
        melodic_contour = row["Melodic Contour"]
        note_density = row["Note Density"]
        dynamic_range = row["Dynamic Range"]
        
        predicted_mood = classify_mood(
            pitch_consistency, duration_consistency, offset_variance, melodic_contour, note_density, dynamic_range, means, stds
        )
        quality_score = evaluate_row_high_level(row, means, stds)
        
        classified_result = {
            "File": row["File"],
            "Predicted Mood": predicted_mood,
            "Melody Score": row["Melody Score"],
            "Harmony Score": row["Harmony Score"],
            "Rhythm Score": row["Rhythm Score"],
            "Overall Structure Score": row["Overall Structure Score"],
            "Quality Score": quality_score
        }
        classified_results.append(classified_result)

    final_results_df = pd.DataFrame(classified_results)

    results_file_path = os.path.join(output_folder, 'midi_evaluation_results_VL.csv')
    if os.path.exists(results_file_path):
        existing_df = pd.read_csv(results_file_path)
        combined_df = pd.concat([existing_df, final_results_df])
        combined_df = combined_df.drop_duplicates(subset=["File"], keep='last')
        combined_df.to_csv(results_file_path, index=False)
        print(f"Results updated in {results_file_path}")
    else:
        final_results_df.to_csv(results_file_path, index=False)
        print(f"Results saved in {results_file_path}")

if __name__ == "__main__":
    main()
