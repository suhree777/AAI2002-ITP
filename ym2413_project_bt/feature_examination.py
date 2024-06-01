import os
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import random

def resample_midi_instruments(midi_path, sample_rate_hz=44500):
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

def plot_instrument_data(times, instrument_data):
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

midi_directory = 'ym2413_project_bt/1_output_limited/Q1_happy'

# Select a random MIDI file from the directory
midi_files = [f for f in os.listdir(midi_directory) if f.endswith('.mid')]
selected_midi = random.choice(midi_files)
midi_path = os.path.join(midi_directory, selected_midi)
print(f"Selected MIDI file for sampling: {selected_midi}")

# Resample MIDI data for each instrument at 50Hz
times, instrument_data = resample_midi_instruments(midi_path)

# Plot the resampled MIDI data for each instrument
plot_instrument_data(times, instrument_data)