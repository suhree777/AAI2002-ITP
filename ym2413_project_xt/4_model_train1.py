import json
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight

def load_data(data_path):
    sequences = []
    labels = []
    mood_labels = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}

    for mood in os.listdir(data_path):
        mood_path = os.path.join(data_path, mood)
        for file in os.listdir(mood_path):
            file_path = os.path.join(mood_path, file)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                for instrument, events in data['instruments'].items():
                    if events:  # ensure there are events
                        sequences.append(events)
                        labels.append(mood_labels[mood])

    # Convert lists to numpy arrays
    sequences = pad_sequences(sequences, padding='post')
    labels = np.array(labels)
    return sequences, labels

def get_max_vocab_size(vocab_directory):
    max_vocab_id = 0

    for filename in os.listdir(vocab_directory):
        filepath = os.path.join(vocab_directory, filename)
        with open(filepath, 'r') as file:
            vocab = json.load(file)
            max_id = max(map(int, vocab.values()))  # Convert values to integers and find the max
            max_vocab_id = max(max_vocab_id, max_id)

    # Since vocab indices are typically 0-based, add 1 to get the correct size
    return max_vocab_id + 1

def build_model(vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    data_path = 'ym2413_project_xt/3_processed_features/data'
    sequences, labels = load_data(data_path)
    print("Loaded sequences:", sequences.shape)
    print("Loaded labels:", labels.shape)

    vocab_directory = 'ym2413_project_xt/3_processed_features/instrument_vocabs'
    max_vocab_size = get_max_vocab_size(vocab_directory)
    print("The maximum vocabulary size is:", max_vocab_size)

    # Hyperparameters
    embedding_dim = 64
    lstm_units = 128
    num_classes = 4  # Number of mood categories
    dropout_rate = 0.4

    model = build_model(max_vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate)
    model.summary()

    model_dir = 'ym2413_project_xt/3_trained_models_1'
    os.makedirs(model_dir, exist_ok=True)

    # Callbacks for monitoring and improving training
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_model.keras'),
        save_best_only=True,  # Saves only the best model
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True  # Restores the best weights found during training upon early stopping
    )

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # Train the model
    history = model.fit(
        sequences, labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2,  # Use 20% of the data for validation
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weights
    )

    # Save the final model to a specified path
    final_model_path = os.path.join(model_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f"Final model saved at {final_model_path}")
