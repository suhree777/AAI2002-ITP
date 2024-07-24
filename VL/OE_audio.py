import librosa
import numpy as np
import os
import pandas as pd

def load_audio(file_path):
    # Load the audio file as a floating point time series.
    audio_data, sampling_rate = librosa.load(file_path, sr=None)
    return audio_data, sampling_rate

def extract_features(audio_data, sampling_rate):
    # Extract basic features
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sampling_rate)
    chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sampling_rate)
    harmonic = librosa.effects.harmonic(audio_data)
    melody = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate)
    return tempo, beats, chroma, harmonic, melody

def evaluate_pitch_consistency(chroma):
    # Evaluate pitch consistency by checking variance in chroma
    return np.var(chroma, axis=1).mean()

def evaluate_temporal_structure(tempo, beats):
    # Temporal structure evaluated by tempo consistency and beat presence
    beat_variability = np.std(np.diff(beats))
    return tempo, beat_variability

def evaluate_harmonicity(harmonic):
    # Harmonicity could be evaluated by the smoothness of the harmonic component
    return librosa.feature.spectral_flatness(y=harmonic).mean()

def evaluate_melodic_contour(melody):
    # Melodic contour can be analyzed by the variability in melodic progression
    melodic_variability = np.var(melody, axis=1).mean()
    return melodic_variability

def evaluate_melody(melody):
    # Assessing the smoothness, hierarchy, and adherence to music theory fundamentals
    melodic_contour = evaluate_melodic_contour(melody)
    return melodic_contour  # Using melodic contour as a proxy

def evaluate_harmony(chroma):
    # Evaluating the coordination and harmony
    harmony_score = np.var(chroma, axis=1).mean()
    return harmony_score

def evaluate_rhythm(tempo, beats):
    # Examining the accuracy and regularity of rhythm
    beat_variability = np.std(np.diff(beats))
    return tempo, beat_variability

def evaluate_overall_structure(audio_data, sampling_rate):
    # Analyzing the tightness, logic, and innovation
    # Using note density and dynamic range as proxies
    dynamic_range = np.ptp(audio_data)
    note_density = len(audio_data) / sampling_rate
    structure_score = note_density + dynamic_range
    return structure_score

def classify_mood(pitch_consistency, tempo, beat_variability, harmonicity, melodic_contour, note_density, dynamic_range):
    if pitch_consistency < 0.5 and beat_variability < 0.1:
        return "Relaxed"
    elif pitch_consistency > 1.5 and beat_variability > 0.3:
        return "Angry"
    elif note_density < 5 and dynamic_range < 30:
        return "Sad"
    elif note_density > 10 and dynamic_range > 40:
        return "Happy"
    else:
        return "Neutral"

def evaluate_music(file_path):
    audio_data, sampling_rate = load_audio(file_path)
    tempo, beats, chroma, harmonic, melody = extract_features(audio_data, sampling_rate)

    # Calculate evaluation metrics
    pitch_consistency = evaluate_pitch_consistency(chroma)
    tempo, beat_variability = evaluate_temporal_structure(tempo, beats)
    harmonicity = evaluate_harmonicity(harmonic)
    melodic_contour = evaluate_melodic_contour(melody)
    melody_score = evaluate_melody(melody)
    harmony_score = evaluate_harmony(chroma)
    rhythm_score = evaluate_rhythm(tempo, beats)
    structure_score = evaluate_overall_structure(audio_data, sampling_rate)
    note_density = len(audio_data) / sampling_rate
    dynamic_range = np.ptp(audio_data)
    predicted_mood = classify_mood(pitch_consistency, tempo, beat_variability, harmonicity, melodic_contour, note_density, dynamic_range)

    return {
        "File": os.path.basename(file_path),
        "Predicted Mood": predicted_mood,
        "Tempo": tempo,
        "Beat Variability": beat_variability,
        "Pitch Consistency": pitch_consistency,
        "Harmonicity": harmonicity,
        "Melodic Contour": melodic_contour,
        "Melody Score": melody_score,
        "Harmony Score": harmony_score,
        "Rhythm Score": rhythm_score,
        "Overall Structure Score": structure_score,
    }

def main():
    print("Current Working Directory:", os.getcwd())
    input_folder = 'VL/1_output'
    output_folder = 'VL/4_evaluation_results'
    results = []

    # Evaluate all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            evaluation_result = evaluate_music(file_path)
            results.append(evaluation_result)
    
    results_df = pd.DataFrame(results)
    
    # Check if the results file already exists
    results_file_path = os.path.join(output_folder, 'audio_evaluation_results.csv')
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
