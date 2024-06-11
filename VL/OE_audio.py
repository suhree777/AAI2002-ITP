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
    return librosa.feature.spectral_flatness(y=harmonic)

def evaluate_melodic_contour(melody):
    # Melodic contour can be analyzed by the variability in melodic progression
    melodic_variability = np.var(melody, axis=1).mean()
    return melodic_variability

def evaluate_music(file_path):
    audio_data, sampling_rate = load_audio(file_path)
    tempo, beats, chroma, harmonic, melody = extract_features(audio_data, sampling_rate)

    # Calculate evaluation metrics
    pitch_consistency = evaluate_pitch_consistency(chroma)
    tempo, beat_variability = evaluate_temporal_structure(tempo, beats)
    harmonicity = evaluate_harmonicity(harmonic).mean()
    melodic_contour = evaluate_melodic_contour(melody)

    return {
        "File": os.path.basename(file_path),
        "Tempo": tempo,
        "Beat Variability": beat_variability,
        "Pitch Consistency": pitch_consistency,
        "Harmonicity": harmonicity,
        "Melodic Contour": melodic_contour
    }

def main():
    input_folder = '1_output'
    output_folder = '3_evaluation_results'
    results = []

    # Evaluate all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            evaluation_result = evaluate_music(file_path)
            results.append(evaluation_result)
    
    # Save the results to a DataFrame and then to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'evaluation_results.csv'), index=False)
    print(f"Results saved in {output_folder}/evaluation_results.csv")

if __name__ == "__main__":
    main()
