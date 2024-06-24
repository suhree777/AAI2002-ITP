import os
import pretty_midi
import numpy as np

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

def find_highest_frequency_in_directory(midi_directory):
    highest_frequency = 0

    midi_files = [f for f in os.listdir(midi_directory) if f.endswith('.mid')]
    for midi_file in midi_files:
        midi_path = os.path.join(midi_directory, midi_file)
        file_highest_frequency = analyze_frequency_content(midi_path)
        if file_highest_frequency > highest_frequency:
            highest_frequency = file_highest_frequency

    return highest_frequency

def resample_midi_instruments(midi_path, sample_rate_hz):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Define the end time based on the last event in the MIDI file
    end_time = midi_data.get_end_time()
    
    # Create a time series from 0 to end_time with intervals based on the sample rate
    times = np.arange(0, end_time, 1/sample_rate_hz)
    
    instrument_data = {}

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            # Initialize arrays to hold interpolated pitch and velocity values
            sampled_pitches = np.zeros_like(times)
            sampled_velocities = np.zeros_like(times)
            
            # Get all note changes and their times
            changes = np.array([[note.start, note.pitch, note.velocity] for note in instrument.notes] +
                               [[note.end, 0, 0] for note in instrument.notes])
            # Sort changes by time
            changes = changes[changes[:, 0].argsort()]
            
            # Current pitch and velocity
            current_pitch = 0
            current_velocity = 0
            
            change_idx = 0
            for i, time in enumerate(times):
                # Update current pitch and velocity based on changes
                while change_idx < len(changes) and changes[change_idx][0] <= time:
                    current_pitch, current_velocity = changes[change_idx][1:3] if changes[change_idx][1] != 0 else (0, 0)
                    change_idx += 1
                sampled_pitches[i] = current_pitch
                sampled_velocities[i] = current_velocity
            
            instrument_data[instrument_name] = (sampled_pitches, sampled_velocities)

    return times, instrument_data

def resample_midi_files(midi_directory, sample_rate_hz):
    midi_files = [f for f in os.listdir(midi_directory) if f.endswith('.mid')]
    for midi_file in midi_files:
        midi_path = os.path.join(midi_directory, midi_file)
        times, instrument_data = resample_midi_instruments(midi_path, sample_rate_hz)
        # Optionally save the resampled MIDI to a new file
        resampled_midi_path = os.path.join(midi_directory, f"resampled_{sample_rate_hz}_{midi_file}")
        midi_data = pretty_midi.PrettyMIDI()
        for instrument_name, (pitches, velocities) in instrument_data.items():
            program = pretty_midi.instrument_name_to_program(instrument_name)
            instrument = pretty_midi.Instrument(program=program)
            for i in range(len(times) - 1):
                if pitches[i] > 0 and velocities[i] > 0:
                    note = pretty_midi.Note(
                        velocity=int(velocities[i]),
                        pitch=int(pitches[i]),
                        start=times[i],
                        end=times[i + 1]
                    )
                    instrument.notes.append(note)
            midi_data.instruments.append(instrument)
        midi_data.write(resampled_midi_path)
        print(f"Resampled {midi_path} to {resampled_midi_path}")

def plot_instrument_data(times, instrument_data):
    import matplotlib.pyplot as plt

    for idx, (instrument_name, (pitches, velocities)) in enumerate(instrument_data.items()):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(2, 1, 1)
        plt.plot(times, pitches, label='Pitch')
        plt.title(f'{instrument_name} - Pitches', fontsize=10)
        plt.ylabel('Pitch', fontsize=10)
        plt.xticks(np.arange(0, times[-1], step=5), fontsize=10)
        plt.yticks(fontsize=10)

        plt.subplot(2, 1, 2)
        plt.plot(times, velocities, label='Velocity', color='r')
        plt.title(f'{instrument_name} - Velocities', fontsize=10)
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Velocity', fontsize=10)
        plt.xticks(np.arange(0, times[-1], step=5), fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        # Set block=True for the last plot
        plt.show(block=(idx == len(instrument_data) - 1))

# Directories for different moods
mood_directories = ['ym2413_project_bt/1_output_limited/Q1_happy',
                    'ym2413_project_bt/1_output_limited/Q2_angry',
                    'ym2413_project_bt/1_output_limited/Q3_sad',
                    'ym2413_project_bt/1_output_limited/Q4_relaxed']

# Find the highest frequency across all mood directories
highest_frequency_overall = 0
for midi_directory in mood_directories:
    highest_frequency = find_highest_frequency_in_directory(midi_directory)
    if highest_frequency > highest_frequency_overall:
        highest_frequency_overall = highest_frequency

print(f"Highest frequency found overall: {highest_frequency_overall} Hz")

# Determine the Nyquist rate and set a common sampling rate
nyquist_rate = 2 * highest_frequency_overall
common_sampling_rate = max(4000, nyquist_rate)  # Use a minimum of 4000 Hz or the calculated Nyquist rate
print(f"Recommended common sampling rate based on Nyquist theorem: {common_sampling_rate} Hz")

# Resample all MIDI files in all mood directories using the common sampling rate
for midi_directory in mood_directories:
    resample_midi_files(midi_directory, common_sampling_rate)

# Select a random MIDI file from one of the directories for demonstration
import random
selected_midi_directory = random.choice(mood_directories)
midi_files = [f for f in os.listdir(selected_midi_directory) if f.endswith('.mid')]
selected_midi = random.choice(midi_files)
midi_path = os.path.join(selected_midi_directory, selected_midi)
print(f"Selected MIDI file for sampling: {selected_midi}")

# Use the recommended common sampling rate for resampling MIDI data
times, instrument_data = resample_midi_instruments(midi_path, sample_rate_hz=common_sampling_rate)

# Plot the resampled MIDI data for each instrument
plot_instrument_data(times, instrument_data)
