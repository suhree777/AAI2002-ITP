import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sequence length
sequence_length = 50

# Load encoded events and vocabulary
def load_data(events_path, vocab_path):
    # Load the vocabulary
    vocab = {}
    with open(vocab_path, 'r') as file:
        for line in file:
            entry, index = line.strip().split('\t')
            event, emotion = eval(entry)  # Convert string tuple to actual tuple
            vocab[int(index)] = (event, emotion)

    # Load the encoded events
    with open(events_path, 'r') as file:
        events = [int(line.strip()) for line in file.readlines()]

    # Create sequences (X) and labels (y)
    X = []
    y = []
    for i in range(len(events) - sequence_length):
        X.append(events[i:i+sequence_length])
        y.append(events[i + sequence_length])

    return np.array(X), np.array(y), vocab

# Prepare dataset
X, y, vocab = load_data('ym2413_project/encoded_events.txt', 'ym2413_project/vocabulary.txt')
Y = to_categorical(y, num_classes=len(vocab))

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=len(vocab), output_dim=100),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(len(vocab), activation='softmax')
])

optimizer = Adam(learning_rate=0.01)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training
checkpoint = ModelCheckpoint('ym2413_project/best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

# Save the final model
model.save('ym2413_project/final_chiptune_music_model.keras')