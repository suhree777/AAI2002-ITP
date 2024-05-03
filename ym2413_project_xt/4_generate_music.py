import pretty_midi
import numpy as np
import os
from keras.models import load_model
import random

def load_trained_model(model_path):
    return load_model(model_path)

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r') as file:
        for line in file:
            event, idx = line.strip().split('\t')
            vocab[event] = int(idx)
    return vocab

def choose_seed_sequence(vocab, sequence_length):
    events = list(vocab.keys())
    if len(events) < sequence_length:
        raise ValueError("Not enough unique events in vocabulary to form a seed sequence.")
    return [vocab[random.choice(events)] for _ in range(sequence_length)]

def generate_music(model, vocab, seed_sequence, num_events_to_generate):
    index_to_event = {index: event for event, index in vocab.items()}
    generated_sequence = seed_sequence[:]
    
    for _ in range(num_events_to_generate):
        input_sequence = np.array(generated_sequence[-len(seed_sequence):]).reshape(1, -1)
        probabilities = model.predict(input_sequence)[0]
        next_event_index = np.random.choice(len(probabilities), p=probabilities)
        generated_sequence.append(next_event_index)

    return [index_to_event[idx] for idx in generated_sequence[len(seed_sequence):]]

def create_midi(generated_events, output_path):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

    current_time = 0
    for event in generated_events:
        if 'NOTEON' in event:
            pitch = int(event.split('_')[2])
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=current_time, end=current_time + 0.5)
            instrument.notes.append(note)
        elif 'NOTEOFF' in event:
            pitch = int(event.split('_')[2])
            for note in instrument.notes:
                if note.pitch == pitch and note.end == current_time:
                    note.end = current_time
                    break
        elif 'WT' in event:
            wait_time = int(event.split('_')[1])
            current_time += wait_time

    midi.instruments.append(instrument)
    midi.write(output_path)

if __name__ == '__main__':
    model_dir = 'ym2413_project/trained_models'
    base_dir = 'ym2413_project/emotion_data'
    music_dir = 'ym2413_project/generated_pcs'
    os.makedirs(music_dir, exist_ok=True)

    emotions = ['Q1_happy', 'Q2_angry', 'Q3_sad', 'Q4_relaxed']
    print("Available Emotions:")
    for i, emotion in enumerate(emotions, 1):
        print(f"{i}. {emotion}")
    selection = int(input("Select an emotion (1-4): ")) - 1
    selected_emotion = emotions[selection]

    model_path = os.path.join(model_dir, f'{selected_emotion}_final_chiptune_music_model.keras')
    vocab_path = os.path.join(base_dir, f'vocabulary_{selected_emotion}.txt')
    
    model = load_trained_model(model_path)
    vocab = load_vocab(vocab_path)
    
    sequence_length = 50
    seed_sequence = choose_seed_sequence(vocab, sequence_length)

    num_events_to_generate = 200
    generated_music = generate_music(model, vocab, seed_sequence, num_events_to_generate)
    print("Generated Music Events:", generated_music)

    # Convert generated music to MIDI
    midi_output_path = os.path.join(music_dir, f'generated_music_{selected_emotion}.mid')
    create_midi(generated_music, midi_output_path)
    print(f"MIDI file created at {midi_output_path}")
