import os
import pandas as pd
import mido
import numpy as np

def load_midi(file_path):
    try:
        return mido.MidiFile(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def evaluate_pitch_consistency(pitches):
    if not pitches:
        return 0
    return np.std(pitches)

def evaluate_temporal_structure(durations, offsets):
    duration_consistency = np.std(durations) if durations else 0
    offset_variance = np.std(offsets) if offsets else 0
    return duration_consistency, offset_variance

def evaluate_melodic_contour(intervals):
    return np.std(intervals) if intervals else 0

def evaluate_note_density_and_dynamic_range(notes, velocities, total_duration):
    note_density = len(notes) / total_duration if total_duration else 0
    dynamic_range = max(velocities) - min(velocities) if velocities else 0
    return note_density, dynamic_range

def evaluate_melody(melodic_contour, pitch_consistency):
    melodic_contour_score = 1 / melodic_contour if melodic_contour else 0
    pitch_consistency_score = 1 / pitch_consistency if pitch_consistency else 0
    return melodic_contour_score + pitch_consistency_score

def evaluate_harmony(chords, total_tracks):
    return chords / total_tracks if total_tracks else 0

def evaluate_rhythm(duration_consistency, offset_variance):
    duration_consistency_score = 1 / duration_consistency if duration_consistency else 0
    offset_variance_score = 1 / offset_variance if offset_variance else 0
    return duration_consistency_score + offset_variance_score

def evaluate_overall_structure(note_density, dynamic_range):
    return note_density + dynamic_range

def normalize_feature(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) if max_value != min_value else 0

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
        return "Angry"
    elif arousal <= 0.5 and valence > 0.5:
        return "Relaxed"
    elif arousal <= 0.5 and valence <= 0.5:
        return "Sad"
    else:
        return "Neutral"

THRESHOLDS = {
    "Melody Score": (0, 2),
    "Harmony Score": (0, 1),
    "Rhythm Score": (0, 2),
    "Overall Structure Score": (0, 2),
}

WEIGHTS = {
    "Melody Score": 2,
    "Harmony Score": 1,
    "Rhythm Score": 1,
    "Overall Structure Score": 2,
}

def evaluate_row(row):
    score = 0
    for metric, (low, high) in THRESHOLDS.items():
        value = row[metric]
        normalized_value = (value - low) / (high - low) if high != low else 0
        normalized_value = max(0, min(normalized_value, 1))
        score += WEIGHTS[metric] * normalized_value
    return score

def evaluate_music_df(df):
    df['Quality Score'] = df.apply(evaluate_row, axis=1)
    return df

def evaluate_music(file_path, selected_features):
    midi_file = load_midi(file_path)
    if not midi_file:
        return None

    pitches, durations, offsets, notes, velocities = [], [], [], [], []
    intervals, time, chords = [], 0, 0

    for track in midi_file.tracks:
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.append(msg.note)
                durations.append(msg.time)
                offsets.append(time)
                notes.append(msg)
                velocities.append(msg.velocity)
                # Improved chord detection
                if len(track) > 1 and any(t.type == 'note_on' for t in track):
                    chords += 1

    intervals = [pitches[i] - pitches[i-1] for i in range(1, len(pitches))]
    total_duration = midi_file.length

    pitch_consistency = evaluate_pitch_consistency(pitches)
    duration_consistency, offset_variance = evaluate_temporal_structure(durations, offsets)
    melodic_contour = evaluate_melodic_contour(intervals)
    note_density, dynamic_range = evaluate_note_density_and_dynamic_range(notes, velocities, total_duration)
    melody_score = evaluate_melody(melodic_contour, pitch_consistency)
    harmony_score = evaluate_harmony(chords, len(midi_file.tracks))
    rhythm_score = evaluate_rhythm(duration_consistency, offset_variance)
    structure_score = evaluate_overall_structure(note_density, dynamic_range)

    predicted_mood = classify_mood(pitch_consistency, duration_consistency, offset_variance, melodic_contour, note_density, dynamic_range)

    result = {
        "File": os.path.basename(file_path),
        "Predicted Mood": predicted_mood,
        "Pitch Consistency": pitch_consistency,
        "Duration Consistency": duration_consistency,
        "Offset Variance": offset_variance,
        "Melodic Contour": melodic_contour,
        "Note Density": note_density,
        "Dynamic Range": dynamic_range,
        "Melody Score": melody_score,
        "Harmony Score": harmony_score,
        "Rhythm Score": rhythm_score,
        "Overall Structure Score": structure_score,
    }

    return result

def main():
    print("Current Working Directory:", os.getcwd())
    input_folder = 'VL/1_output/samples'
    output_folder = 'VL/4_evaluation_results'
    selected_features = ['melody', 'harmony', 'rhythm', 'overall_structure']
    results = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    midi_files = [f for f in os.listdir(input_folder) if f.endswith('.mid') or f.endswith('.midi')]

    for filename in midi_files:
        file_path = os.path.join(input_folder, filename)
        evaluation_result = evaluate_music(file_path, selected_features)
        if evaluation_result:
            results.append(evaluation_result)

    results_df = pd.DataFrame(results)
    results_df = evaluate_music_df(results_df)

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
