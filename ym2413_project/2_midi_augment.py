import pretty_midi
import os
import numpy as np
import random

def transpose_midi(midi_data, semitones):
    """ Transpose all notes in a MIDI data by a number of semitones, within MIDI range. """
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            new_pitch = note.pitch + semitones
            if 0 <= new_pitch <= 127:  # Ensure the pitch is within a valid MIDI range
                note.pitch = new_pitch
    return midi_data

def augment_midi_file(file_path, output_folder, num_copies):
    """ Augment a single MIDI file by creating multiple copies with random transpositions, maintaining original tempo. """
    base_midi = pretty_midi.PrettyMIDI(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(num_copies):
        transpose_semitones = random.randint(-4, 4)  # Random transpose between -4 and +4 semitones
        new_midi = transpose_midi(base_midi, transpose_semitones)
        new_file_name = f"{base_name}_aug{i}_trans{transpose_semitones}.mid"
        new_midi.write(os.path.join(output_folder, new_file_name))

def augment_folder(input_folder, output_folder, target_count):
    """ Augment all MIDI files in a folder to ensure the folder reaches the target count of files. """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    existing_files = len(files)
    augmentations_per_file = max(1, (target_count - existing_files) // existing_files)

    for file in files:
        if file.lower().endswith('.mid'):
            file_path = os.path.join(input_folder, file)
            augment_midi_file(file_path, output_folder, augmentations_per_file)

    # Handle any remaining files if needed due to integer division
    remaining_augmentations = target_count - len(os.listdir(output_folder))
    if remaining_augmentations > 0:
        for i in range(remaining_augmentations):
            file = random.choice(files)
            if file.lower().endswith('.mid'):
                file_path = os.path.join(input_folder, file)
                augment_midi_file(file_path, output_folder, 1)

def process_all_folders(root_folder, output_root, target_count=228):
    """ Process all sub-folders in the root folder to ensure each has enough files. """
    for sub_folder in os.listdir(root_folder):
        full_input_path = os.path.join(root_folder, sub_folder)
        full_output_path = os.path.join(output_root, sub_folder)
        if os.path.isdir(full_input_path):
            augment_folder(full_input_path, full_output_path, target_count)


if __name__ == '__main__':
    root_folder = 'ym2413_project/output'
    output_root = 'ym2413_project/output_with_generation'
    process_all_folders(root_folder, output_root, 228)

    print("Data augmentation complete.")
