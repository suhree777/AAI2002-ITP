import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Sequence length
sequence_length = 50

def load_data(encoded_events_path, vocab_path):
    # Load vocabulary
    vocab = {}
    with open(vocab_path, 'r') as file:
        for idx, line in enumerate(file):
            event = line.strip()
            vocab[event] = idx

    # Load encoded events
    with open(encoded_events_path, 'r') as file:
        encoded_events = [int(line.strip()) for line in file.readlines()]

    # Create sequences
    X = []
    y = []
    for i in range(len(encoded_events) - sequence_length):
        sequence = encoded_events[i:i + sequence_length]
        label = encoded_events[i + sequence_length]
        X.append(sequence)
        y.append(label)

    X = np.array(X)
    y = to_categorical(y, num_classes=len(vocab))
    return X, y, vocab

def build_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=sequence_length),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(vocab_size, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model_for_emotion(X, y, emotion, output_dir, model_dir):
    vocab_size = y.shape[1]  # Automatically determine the vocabulary size from the output classes
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(vocab_size)

    # Ensure the directory for saving models exists
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, f'{emotion}_best_model.keras')
    final_model_path = os.path.join(model_dir, f'{emotion}_final_chiptune_music_model.keras')

    # Setup checkpointing and early stopping
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # Training the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

    # Save the final model
    model.save(final_model_path)
    print(f"Model for {emotion} saved to {final_model_path}")

    return history

if __name__ == '__main__':
    base_dir = 'ym2413_project/emotion_data'
    model_dir = 'ym2413_project/trained_models'

    emotions = ['Q1_happy', 'Q2_angry', 'Q3_sad', 'Q4_relaxed']
    histories = {}  # Dictionary to store history objects for each emotion

    for emotion in emotions:
        encoded_events_path = os.path.join(base_dir, f'encoded_events_{emotion}.txt')
        vocab_path = os.path.join(base_dir, f'vocabulary_{emotion}.txt')
        X, y, vocab = load_data(encoded_events_path, vocab_path)

        history = train_model_for_emotion(X, y, emotion, base_dir, model_dir)
        histories[emotion] = history  # Store the history for each emotion
