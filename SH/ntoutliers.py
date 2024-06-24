import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import random

# Analyse MIDI files to find highest frequency
def analyze_frequency_content(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    highest_frequency = 0

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                # MIDI note number to frequency conversion
                frequency = pretty_midi.note_number_to_hz(note.pitch)
                if frequency > highest_frequency:
                    highest_frequency = frequency
    
    return highest_frequency

def find_highest_frequencies_in_directory(midi_directory):
    frequencies = []

    midi_files = [f for f in os.listdir(midi_directory) if f.endswith('.mid')]
    for midi_file in midi_files:
        midi_path = os.path.join(midi_directory, midi_file)
        highest_frequency = analyze_frequency_content(midi_path)
        frequencies.append((midi_file, highest_frequency))

    return frequencies

# Detemrine sample rate for each file based on Nyquist Theorem

def calculate_sample_rates(frequencies):
    sample_rates = [(midi_file, 2 * frequency) for midi_file, frequency in frequencies]
    return sample_rates

# Identify and Display outlers

def identify_outliers(sample_rates, threshold=1.5):
    rates = [rate for _, rate in sample_rates]
    median = np.median(rates)
    q1 = np.percentile(rates, 25)
    q3 = np.percentile(rates, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = [(midi_file, rate) for midi_file, rate in sample_rates if rate < lower_bound or rate > upper_bound]
    return outliers

def plot_sample_rates(sample_rates, outliers, directory):
    files, rates = zip(*sample_rates)
    outlier_files, outlier_rates = zip(*outliers) if outliers else ([], [])

    plt.figure(figsize=(12, 6))
    plt.plot(files, rates, 'bo-', label='Sample Rate')
    plt.plot(outlier_files, outlier_rates, 'ro', label='Outliers')
    plt.axhline(y=np.median(rates), color='g', linestyle='--', label='Median')
    plt.xticks(rotation=90)
    plt.xlabel('MIDI Files')
    plt.ylabel('Sample Rate (Hz)')
    plt.title(f'Sample Rates for Each MIDI File in {directory} with Outliers Highlighted')
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_directory(midi_directory):
    print(f"\nProcessing directory: {midi_directory}")
    frequencies = find_highest_frequencies_in_directory(midi_directory)
    sample_rates = calculate_sample_rates(frequencies)
    outliers = identify_outliers(sample_rates)

    print(f"Sample Rates for Each MIDI File in {midi_directory}:")
    for midi_file, rate in sample_rates:
        print(f"{midi_file}: {rate} Hz")

    print("\nOutliers:")
    for midi_file, rate in outliers:
        print(f"{midi_file}: {rate} Hz")
    
    plot_sample_rates(sample_rates, outliers, midi_directory)

def main():
    directories = [
        'ym2413_project_bt/1_output_limited/Q1_happy',
        'ym2413_project_bt/1_output_limited/Q2_angry',
        'ym2413_project_bt/1_output_limited/Q3_sad',
        'ym2413_project_bt/1_output_limited/Q4_relaxed'
    ]

    for midi_directory in directories:
        process_directory(midi_directory)

# Run the analysis and plotting for all directories
main()
