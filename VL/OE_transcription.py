import os
import pandas as pd
import mido
import numpy as np

def load_midi(file_path):
    """Load a MIDI file and handle errors gracefully."""
    try:
        return mido.MidiFile(file_path)
    except Exception as e:
        print(f"Error loading MIDI file {file_path}: {e}")
        return None

def evaluate_pitch_consistency(pitches):
    """Evaluate pitch consistency by calculating the standard deviation of pitch counts."""
    return pd.Series(pitches).value_counts().std() if len(pitches) > 0 else 0

def evaluate_temporal_structure(durations, offsets):
    """Evaluate temporal structure by calculating duration consistency and offset variance."""
    if not durations or not offsets:
        return 0, 0
    duration_consistency = pd.Series(durations).value_counts().std()
    offset_variance = pd.Series(offsets).diff().abs().mean()
    return duration_consistency, offset_variance

def evaluate_melodic_contour(intervals):
    """Evaluate melodic contour by calculating the mean absolute difference of intervals."""
    return pd.Series(intervals).diff().abs().mean() if len(intervals) > 0 else 0

def evaluate_note_density_and_dynamic_range(notes, velocities, total_duration):
    """Evaluate note density and dynamic range."""
    note_density = len(notes) / total_duration if total_duration else 0
    dynamic_range = max(velocities) - min(velocities) if velocities else 0
    return note_density, dynamic_range

def evaluate_melody(melodic_contour, pitch_consistency):
    """Evaluate melody by combining melodic contour and pitch consistency."""
    return (1 / melodic_contour if melodic_contour != 0 else 0) + (1 / pitch_consistency if pitch_consistency != 0 else 0)

def evaluate_harmony(chords, total_tracks):
    """Evaluate harmony by the ratio of chords to total tracks."""
    return chords / total_tracks if total_tracks != 0 else 0

def evaluate_rhythm(duration_consistency, offset_variance):
    """Evaluate rhythm by combining duration consistency and offset variance."""
    return (1 / duration_consistency if duration_consistency != 0 else 0) + (1 / offset_variance if offset_variance != 0 else 0)

def evaluate_overall_structure(note_density, dynamic_range):
    """Evaluate overall structure by combining note density and dynamic range."""
    return note_density + dynamic_range

def normalize_feature(value, mean, std):
    """Normalize a feature value."""
    return (value - mean) / std if std != 0 else 0

def scale_to_100(normalized_value):
    """Scale a normalized value to the range [0, 100]."""
    return max(0, min(50 + 50 * normalized_value, 100))

def classify_mood(features, means, stds):
    """Classify mood based on normalized features."""
    norm_features = {key: normalize_feature(value, means[key], stds[key]) for key, value in features.items()}
    
    arousal = sum(norm_features.values()) / len(norm_features)
    valence = sum(1 - v for v in norm_features.values()) / len(norm_features)

    print(f"arousal: {arousal}, valence: {valence}")

    if arousal > 0 and valence > 0:
        return "Happy"
    elif arousal > 0 and valence <= 0:
        return "Angry"
    elif arousal <= 0 and valence > 0:
        return "Relaxed"
    elif arousal <= 0 and valence <= 0:
        return "Sad"
    return "Neutral"

def impute_missing_values(df, metric):
    """Impute missing values in a dataframe column with the mean."""
    df[metric] = df[metric].fillna(df[metric].mean())
    return df

def calculate_mean_std(df, metric):
    """Calculate mean and standard deviation of a dataframe column."""
    return df[metric].mean(), df[metric].std()

WEIGHTS = {
    "Melody Score": 2,
    "Harmony Score": 1,
    "Rhythm Score": 1,
    "Overall Structure Score": 2
}

def evaluate_row_high_level(row, means, stds):
    """Evaluate high-level quality score for a row of metrics."""
    score = 0
    max_score = 0
    for metric, weight in WEIGHTS.items():
        value = row[metric]
        if value is not None:
            normalized_value = normalize_feature(value, means[metric], stds[metric])
            scaled_value = scale_to_100(normalized_value)
            score += weight * scaled_value
            max_score += weight * 100
    return round((score / max_score) * 100, 3) if max_score != 0 else 0

def evaluate_music_df_high_level(df, means, stds):
    """Evaluate high-level quality score for a dataframe of metrics."""
    df['Quality Score'] = df.apply(evaluate_row_high_level, axis=1, means=means, stds=stds)
    return df

def evaluate_music(file_path):
    """Evaluate various musical features from a MIDI file."""
    midi_file = load_midi(file_path)
    if midi_file is None:
        return None

    pitches, durations, offsets, notes, velocities = [], [], [], [], []
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

    intervals = np.diff(pitches) if len(pitches) > 1 else []
    total_duration = midi_file.length

    pitch_consistency = evaluate_pitch_consistency(pitches)
    duration_consistency, offset_variance = evaluate_temporal_structure(durations, offsets)
    melodic_contour = evaluate_melodic_contour(intervals)
    note_density, dynamic_range = evaluate_note_density_and_dynamic_range(notes, velocities, total_duration)

    melody_score = evaluate_melody(melodic_contour, pitch_consistency)
    harmony_score = evaluate_harmony(chords, len(midi_file.tracks))
    rhythm_score = evaluate_rhythm(duration_consistency, offset_variance)
    structure_score = evaluate_overall_structure(note_density, dynamic_range)

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

    print(f"Evaluated features for {file_path}: {feature_values}")
    return feature_values

def main():
    input_folder = 'ym2413_project_bt/1_output_freq/Q1_happy'
    #input_folder = 'VL/1_output'
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

    for metric in ["Pitch Consistency", "Duration Consistency", "Offset Variance", "Melodic Contour", "Note Density", "Dynamic Range",
                   "Melody Score", "Harmony Score", "Rhythm Score", "Overall Structure Score"]:
        results_df = impute_missing_values(results_df, metric)

    means, stds = {}, {}
    for metric in ["Pitch Consistency", "Duration Consistency", "Offset Variance", "Melodic Contour", "Note Density", "Dynamic Range",
                   "Melody Score", "Harmony Score", "Rhythm Score", "Overall Structure Score"]:
        means[metric], stds[metric] = calculate_mean_std(results_df, metric)

    classified_results = []
    for index, row in results_df.iterrows():
        features = {
            "Pitch Consistency": row["Pitch Consistency"],
            "Duration Consistency": row["Duration Consistency"],
            "Offset Variance": row["Offset Variance"],
            "Melodic Contour": row["Melodic Contour"],
            "Note Density": row["Note Density"],
            "Dynamic Range": row["Dynamic Range"]
        }

        predicted_mood = classify_mood(features, means, stds)
        quality_score = evaluate_row_high_level(row, means, stds)
        
        classified_result = {
            "File": row["File"],
            "Russell's Mood": predicted_mood,
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
