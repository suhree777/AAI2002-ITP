import pretty_midi
import os

def calculate_sample_rate(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    min_delta = float('inf')
    
    for instrument in midi_data.instruments:
        notes = sorted(instrument.notes, key=lambda note: note.start)
        for i in range(1, len(notes)):
            delta = notes[i].start - notes[i-1].start
            if delta > 0:
                min_delta = min(min_delta, delta)
    
    if min_delta == float('inf'):
        return 44100  # Default standard sample rate for audio files
    
    frequency = 1 / min_delta
    sample_rate = int(frequency * 2 * 2)  # Nyquist rate with a margin
    return sample_rate

def analyze_folder_for_highest_sample_rate(folder_path):
    highest_sample_rate = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mid', '.midi')):  # Checks for both .mid and .midi files
            file_path = os.path.join(folder_path, filename)
            sample_rate = calculate_sample_rate(file_path)
            highest_sample_rate = max(highest_sample_rate, sample_rate)

    # Display the highest sample rate found
    print(f"The highest sample rate suggested across all MIDI files: {highest_sample_rate} Hz")
    return highest_sample_rate


# Example usage
midi_path = 'ym2413_project_bt/1_output_limited/Q4_relaxed/'
highest_sample_rate = analyze_folder_for_highest_sample_rate(midi_path)
