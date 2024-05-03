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

def choose_seed_sequence(vocab, sequence_length, categories=None):
    """
    Randomly choose a seed sequence from the vocabulary, ensuring diversity by selecting from different categories.
    """
    if categories is None:
        # Simple categorisation based on the instrument in the event name
        categories = {
            'P1': [event for event in vocab.keys() if 'P1_' in event],
            'P2': [event for event in vocab.keys() if 'P2_' in event],
            'TR': [event for event in vocab.keys() if 'TR_' in event],
            'NO': [event for event in vocab.keys() if 'NO_' in event]
        }

    selected_events = []
    remaining_length = sequence_length
    
    # Ensure at least one event from each category if possible
    for category, events in categories.items():
        if events:  # Ensure there are events in this category
            selected_event = random.choice(events)
            selected_events.append(selected_event)
            remaining_length -= 1
    
    # Fill the rest of the sequence with random choices
    all_events = sum(categories.values(), [])
    selected_events.extend(random.choices(all_events, k=remaining_length))

    # Map selected event names to their indices
    return [vocab[event] for event in selected_events]

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

    num_events_to_generate = 200
    generated_music = generate_music(model, vocab, seed_sequence, num_events_to_generate)
    print("Generated Music Events:", generated_music)

    # Convert generated music to MIDI
    midi_output_path = os.path.join(music_dir, f'generated_music_{selected_emotion}.mid')
    create_midi(generated_music, midi_output_path)
    print(f"MIDI file created at {midi_output_path}")
