import os
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import random

def resample_midi_instruments(midi_path, sample_rate_hz=500):
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
    for instrument_name, (pitches, velocities) in instrument_data.items():
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(times, pitches, label='Pitch')
        plt.title(f'{instrument_name} - Pitches')
        plt.ylabel('Pitch')
        
        plt.subplot(2, 1, 2)
        plt.plot(times, velocities, label='Velocity', color='r')
        plt.title(f'{instrument_name} - Velocities')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        
        plt.tight_layout()
        plt.show(block=False)


midi_directory = 'ym2413_project_bt\output\Q1_happy'

# Select a random MIDI file from the directory
midi_files = [f for f in os.listdir(midi_directory) if f.endswith('.mid')]
selected_midi = random.choice(midi_files)
midi_path = os.path.join(midi_directory, selected_midi)

# Resample MIDI data for each instrument at 50Hz
times, instrument_data = resample_midi_instruments(midi_path)

# Plot the resampled MIDI data for each instrument
plot_instrument_data(times, instrument_data)
user_input = input("Please enter something: ")