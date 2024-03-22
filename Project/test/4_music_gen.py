import numpy as np
import pretty_midi
from tensorflow import keras
from keras.models import load_model
import random

def generate_sequence(model, seed_sequence, target_duration, vocab_size, inv_vocab):
    generated_sequence = seed_sequence.copy()
    current_duration = 0
    while current_duration < target_duration:
        input_sequence = np.array([generated_sequence[-50:]])
        predicted_event = model.predict(input_sequence)[0]
        predicted_event_index = np.argmax(predicted_event)
        generated_sequence.append(predicted_event_index)
        event = inv_vocab[predicted_event_index]
        if event.startswith('WT_'):
            wait_time = int(event.split('_')[1]) / 100
            current_duration += wait_time
    return generated_sequence

def sequence_to_midi(sequence, inv_vocab, output_file_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    current_time = 0
    current_notes = {}
    for event_index in sequence:
        event = inv_vocab[event_index]
        if event.startswith('NOTEON_'):
            pitch = int(event.split('_')[1])
            current_notes[pitch] = current_time
        elif event.startswith('NOTEOFF_'):
            pitch = int(event.split('_')[1])
            if pitch in current_notes:
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=current_notes[pitch], end=current_time)
                instrument.notes.append(note)
                del current_notes[pitch]
        elif event.startswith('WT_'):
            wait_time = int(event.split('_')[1]) / 100
            current_time += wait_time
    midi.instruments.append(instrument)
    midi.write(output_file_path)

if __name__ == '__main__':
    model = load_model('model/lstm_model.h5')

    with open('preprocess/vocabulary.txt', 'r') as f:
        vocab = {line.split('\t')[0].strip(): int(line.split('\t')[1]) for line in f.readlines()}
    inv_vocab = {v: k for k, v in vocab.items()}

    with open('preprocess/encoded_events.txt', 'r') as f:
        encoded_events = [int(line.strip()) for line in f.readlines()]

    seed_start_index = random.randint(0, len(encoded_events) - 51)
    seed_sequence = encoded_events[seed_start_index:seed_start_index + 50]

    target_duration = 5 # in seconds
    generated_sequence = generate_sequence(model, seed_sequence, target_duration, len(vocab), inv_vocab)
    sequence_to_midi(generated_sequence, inv_vocab, 'test/generated_music2.mid')
