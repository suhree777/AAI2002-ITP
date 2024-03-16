import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import Adam


def train_lstm_model(encoded_events_file, vocabulary_file, model_save_path):
    # Load the encoded events and vocabulary
    print("Loading encoded events and vocabulary...")
    with open(encoded_events_file, 'r') as f:
        encoded_events = [int(line.strip()) for line in f.readlines()]

    with open(vocabulary_file, 'r') as f:
        vocab = {line.split('\t')[0].strip(): int(line.split('\t')[1]) for line in f.readlines()}

    # Prepare the data
    print("Preparing data...")
    sequence_length = 50
    input_sequences = []
    output_sequences = []
    for i in range(len(encoded_events) - sequence_length):
        input_sequences.append(encoded_events[i:i + sequence_length])
        output_sequences.append(encoded_events[i + sequence_length])

    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)

    # Define the LSTM model
    print("Defining LSTM model...")
    vocab_size = len(vocab)
    lstm_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, lstm_units, input_length=sequence_length))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train the model
    print("Training model...")
    model.fit(input_sequences, output_sequences, epochs=10, batch_size=64, validation_split=0.2)

    # Save the trained model
    print("Saving trained model...")
    model.save(model_save_path)
    print(f"Model saved as {model_save_path}")


if __name__ == '__main__':
    # Set a random seed for reproducibility (Not tested)
    # np.random.seed(42)
    # tf.random.set_seed(42)

    encoded_events_file = 'preprocess/encoded_events.txt'
    vocabulary_file = 'preprocess/vocabulary.txt'
    model_save_path = 'model/lstm_model.h5'
    train_lstm_model(encoded_events_file, vocabulary_file, model_save_path)
