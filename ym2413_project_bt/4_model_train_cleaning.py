import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping

moods = ['happy', 'angry', 'sad', 'relaxed'] # Adjust as per your moods

def mood_to_onehot(mood):
    one_hot = np.zeros(len(moods))
    index = moods.index(mood)
    one_hot[index] = 1
    return one_hot

def load_and_normalize_data(filepath, instrument_name):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    if instrument_name not in data['instruments']:
        return None  # Skip files where the instrument is not present

    # Extract mood and instrument_vector
    mood_vector = mood_to_onehot(data['mood'])
    instrument_vector = data['instrument_vector']
    feature_vector = np.concatenate([mood_vector, instrument_vector])
    
    # Extract instrument data
    instrument_data = np.array(data['instruments'][instrument_name])
    final_input = np.concatenate([feature_vector, instrument_data])
    # print(f"Processed data for {instrument_name}: {instrument_data}")
    return final_input

def process_instrument_data(root_dir, instrument_name):
    all_data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(subdir, file)
                try:
                    processed_data = load_and_normalize_data(filepath, instrument_name)
                    all_data.append(processed_data)
                except Exception as e:
                    print(f"Failed to process {filepath}: {str(e)}")
    return all_data

def create_dataset(sequences, window_size):
    X = []
    y = []
    for data in sequences:
        for i in range(len(data) - window_size):
            seq_x, seq_y = data[i:(i + window_size)], data[i + window_size]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

for i in range(20):
    print("\n")

instrument_vocab_path = 'ym2413_project_bt/3_processed_feature_limited/instrument_list.json'
with open(instrument_vocab_path, 'r') as f:
    instrument_list = json.load(f)

# Setting up early stopping based on accuracy
early_stopping = EarlyStopping(
    monitor='accuracy',        # Monitor the training accuracy
    min_delta=0.01,            # Minimum change to qualify as an improvement
    patience=4,                # Allow slight fluctuations for 3 epochs
    verbose=1,                 # Print messages when stopping
    mode='max',                # We aim to maximize accuracy
    restore_best_weights=True  # Restore model weights from the epoch with the best accuracy
)

for instrument_name in instrument_list:
    directory_path = 'ym2413_project_bt/3_processed_feature_limited/data/'
    vocab_file_path = os.path.join('ym2413_project_bt/3_processed_feature_limited/instrument_vocabs/', f"{instrument_name}_vocab.json")

    instrument_datasets = process_instrument_data(directory_path, instrument_name)
    # print(f"instrument_datasets data for {instrument_datasets[:4]}")
    print(f"instrument_datasets data: {len(instrument_datasets)} samples")

    def clean_data(data):
        return [x for x in data if x is not None]

    instrument_datasets = clean_data(instrument_datasets)
    # print(f"Cleaned instrument_datasets data: {instrument_datasets[:4]}")
    print(f"Cleaned instrument_datasets data: {len(instrument_datasets)} samples")

    window_size = 25
    X, y = create_dataset(instrument_datasets, window_size)

    with open(vocab_file_path, 'r') as f:
        vocab = json.load(f)
    token_size = len(vocab) + len(instrument_list) + len(moods)  # The total number of unique tokens
    print(f"Loaded vocabulary for {instrument_name} with size: {token_size}")

    embedding_dim = 256

    model = Sequential([
        Embedding(token_size, embedding_dim, input_length=window_size),
        LSTM(256, return_sequences=False),
        Dense(token_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X, y, epochs=80, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

    # Saving the model
    mode_path = os.path.join('ym2413_project_bt/model_folder', f"{instrument_name}_model.keras")
    model.save(mode_path)
    print(f'Model {instrument_name}_model.keras saved to ym2413_project_bt/model_folder')