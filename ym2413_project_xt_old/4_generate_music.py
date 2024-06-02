import pretty_midi
import numpy as np
import os
import time
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
    categories = {
        'P1': [e for e in events if 'P1_' in e],
        'P2': [e for e in events if 'P2_' in e],
        'TR': [e for e in events if 'TR_' in e],
        'NO': [e for e in events if 'NO_' in e]
    }

    # Shuffle events within each category
    for category in categories:
        random.shuffle(categories[category])

    # Distribute the selection evenly across categories
    selected_events = []
    while len(selected_events) < sequence_length:
        for category in categories:
            if categories[category] and len(selected_events) < sequence_length:
                # Add one event from each category in a round-robin fashion
                selected_events.append(categories[category].pop(0))

    # Ensure the sequence is exactly the required length
    selected_events = selected_events[:sequence_length]
    return [vocab[event] for event in selected_events]

def generate_music(model, vocab, seed_sequence, num_events_to_generate):
    index_to_event = {index: event for event, index in vocab.items()}
    generated_sequence = seed_sequence[:]
    
    for _ in range(num_events_to_generate):
        input_sequence = np.array(generated_sequence[-len(seed_sequence):]).reshape(1, -1)
        probabilities = model.predict(input_sequence)[0]
        # Sample from the distribution rather than taking the max
        next_event_index = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        generated_sequence.append(next_event_index)

    return [index_to_event[idx] for idx in generated_sequence[len(seed_sequence):]]

def create_midi(generated_events, output_path, total_seconds=15):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    current_time = 0
    note_duration = total_seconds / len(generated_events)  # Adjust duration based on number of events
    for event in generated_events:
        if 'NOTEON' in event:
            pitch = int(event.split('_')[2])
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=current_time,
                end=current_time + note_duration)
            instrument.notes.append(note)
        current_time += note_duration
    midi.instruments.append(instrument)
    midi.write(output_path)

if __name__ == '__main__':
    model_dir = 'ym2413_project_xt/trained_models'
    base_dir = 'ym2413_project_xt/emotion_data'
    music_dir = 'ym2413_project_xt/generated_pcs'
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

    num_events_to_generate = 100
    generated_music = generate_music(model, vocab, seed_sequence, num_events_to_generate)
    print("Generated Music Events:", generated_music)

    # Generating unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    midi_output_path = os.path.join(music_dir, f'generated_music_{selected_emotion}_{timestamp}.mid')
    create_midi(generated_music, midi_output_path, total_seconds=15)
    print(f"MIDI file created at {midi_output_path}")
