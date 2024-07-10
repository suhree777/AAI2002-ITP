import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape, TimeDistributed, Embedding, Input, LSTM, Dropout, Dense, Concatenate, Lambda, Attention
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def mood_to_onehot(mood):
    one_hot = np.zeros(len(moods))
    index = moods.index(mood)
    one_hot[index] = 1
    return one_hot

def load_and_normalize_all_instruments(filepath, instrument_list):
    with open(filepath, 'r') as file:
        data = json.load(file)

    mood_vector = mood_to_onehot(data['mood'])
    instrument_vector = data['instrument_vector']
    feature_vector = np.concatenate([mood_vector, instrument_vector])
    
    all_instrument_data = []
    max_length = 0
    
    # Determine the maximum sequence length and collect data in a single pass
    instrument_data_arrays = {}
    for instrument in instrument_list:
        if instrument in data['instruments']:
            instrument_data = np.array(data['instruments'][instrument])
            max_length = max(max_length, len(instrument_data))
            instrument_data_arrays[instrument] = instrument_data
        else:
            instrument_data_arrays[instrument] = np.array([])  # Use an empty array for missing data

    # Pad instrument data
    for instrument in instrument_list:
        instrument_data = instrument_data_arrays[instrument]
        if instrument_data.size > 0:
            padded_data = np.pad(instrument_data, (0, max_length - len(instrument_data)), 'constant')
        else:
            padded_data = np.zeros(max_length)  # Pad with zeros if instrument data is entirely missing
        all_instrument_data.append(padded_data)

    combined_instrument_data = np.stack(all_instrument_data, axis=-1)
    # print(f"Processed data from {filepath}\nFeature Vector Shape: {feature_vector.shape}, Instrument Data Shape: {combined_instrument_data.shape}")
    return feature_vector, combined_instrument_data

def create_sliding_window_dataset(feature_vector, instrument_data, window_length, target_instrument_index):
    X = []
    y = []
    num_instruments = instrument_data.shape[1]
    sequence_length = instrument_data.shape[0]

    # Print initial data for clarity
    # print("Initial instrument data shape:", instrument_data.shape)
    # print("Number of instruments:", num_instruments)
    # print("Sequence length:", sequence_length)

    # Initial prediction from empty set
    initial_feature_vector = np.concatenate([feature_vector] + [np.zeros(window_length) for _ in range(num_instruments)])
    initial_targets = instrument_data[0, target_instrument_index]  # First actual value for the target instrument
    X.append(initial_feature_vector)
    y.append(initial_targets)

    # Calculate how many windows can be formed
    for start in range(sequence_length - window_length):
        end = start + window_length
        
        # Initialize a list to hold the sliding window for each instrument
        windows = []
        
        # Iterate through each instrument to extract its specific window
        for i in range(num_instruments):
            # Extract the window for the current instrument
            instrument_window = instrument_data[start:end, i]
            windows.append(instrument_window)
        
        # Concatenate all instrument windows and the feature vector to form the full sequence
        full_sequence = np.concatenate([feature_vector] + windows)
        
        # Define target outputs: the next value only for the target instrument
        if end < sequence_length:
            target = instrument_data[end, target_instrument_index]
            X.append(full_sequence)
            y.append(target)

    return np.array(X), np.array(y)

"""def create_sliding_window_dataset(feature_vector, instrument_data, window_length):
    X = []
    y = []
    num_instruments = instrument_data.shape[1]
    sequence_length = instrument_data.shape[0]

    # Print initial shapes and data
    print("\n")
    print("Initial instrument data shape:", instrument_data.shape)
    print("Number of instruments:", num_instruments)
    print("Sequence length:", sequence_length)

    # Initial prediction from empty set
    initial_feature_vector = np.concatenate([feature_vector] + [np.zeros(window_length) for _ in range(num_instruments)])
    initial_targets = instrument_data[0, :]  # Targets for the initial prediction are the first actual values
    X.append(initial_feature_vector)
    y.append(initial_targets)

    print("Initial feature vector shape:", initial_feature_vector.shape)
    print("Initial targets shape:", initial_targets.shape)

    # Calculate how many windows can be formed
    for start in range(sequence_length - window_length):
        end = start + window_length
        
        # Initialize a list to hold the sliding window for each instrument
        windows = []
        
        # Iterate through each instrument to extract its specific window
        for i in range(num_instruments):
            # Extract the window for the current instrument
            instrument_window = instrument_data[start:end, i]
            windows.append(instrument_window)
        
        # Concatenate all instrument windows and the feature vector to form the full sequence
        full_sequence = np.concatenate([feature_vector] + windows)
        
        # Define target outputs: the next values for each instrument
        if end < sequence_length:
            targets = instrument_data[end, :]
            X.append(full_sequence)
            y.append(targets)

        # Print shapes and details about the process
        if start % 1000 == 0:  # Print details every 1000 steps to avoid too much output
            print("\n")
            print(f"At start index {start}:")
            print(f"Window {np.array(windows)}and shape {np.array(windows).shape}")
            print(f"Full sequence {full_sequence} and shape {full_sequence.shape}")
            print(f"Targets {targets} and shape {targets.shape}")
            print("\n")

    return np.array(X), np.array(y)"""

def process_dataset(root_dir, instrument_list, window_length, index):
    all_X = []
    all_y = []

    for file in os.listdir(root_dir):
        filepath = os.path.join(root_dir, file)
        feature_vector, instrument_data = load_and_normalize_all_instruments(filepath, instrument_list)
        X, y = create_sliding_window_dataset(feature_vector, instrument_data, window_length, index)
        all_X.extend(X)
        all_y.extend(y)
    print("All eligable data has been processed")
    return len(feature_vector), np.array(all_X), np.array(all_y)

for i in range(20):
    print("\n")

directory_path = 'ym2413_project_bt/3_processed_freq/'
model_folder = 'ym2413_project_bt/model_folder_freq'
summary_path = os.path.join(directory_path, 'summary.json')
with open(summary_path, 'r') as f:
    data = json.load(f)
    instrument_list = data['top_instrument_names']
    moods = data['mood_labels']
    sample_rate = data['desired_sample_rate']

print(f"Summary file list has been loaded")
print(f"Desired sample rate: {sample_rate}")
print(f"Mood labels: {moods}")
print(f"Instrument list of {len(instrument_list)}: {instrument_list}")

vocab_file_path = os.path.join(directory_path, 'event_vocab.json')
with open(vocab_file_path, 'r') as f:
    vocab = json.load(f)

token_size = len(vocab) + len(instrument_list) + len(moods)  # The total number of unique tokens
# Setting up early stopping based on accuracy
early_stopping = EarlyStopping(
    monitor='accuracy',        # Monitor the training accuracy
    min_delta=0.005,            # Minimum change to qualify as an improvement
    patience=7,                # Allow slight fluctuations for 3 epochs
    verbose=1,                 # Print messages when stopping
    mode='max',                # We aim to maximize accuracy
    restore_best_weights=True  # Restore model weights from the epoch with the best accuracy
)

window_length = sample_rate # Sliding window size per instrment
embedding_dim = 128
num_epoch = 5
batch_size = 512

for index, instrument_name in enumerate(instrument_list):

    num_features, X, y = process_dataset(os.path.join(directory_path, 'data/'), instrument_list, window_length, index)
    print("\n")
    print(f"Summary of settings and inputs")
    print(f"Desired sample rate: {sample_rate}")
    print(f"Mood labels: {moods}")
    print(f"Instrument list of {len(instrument_list)}: {instrument_list} \n")
    print(f"Now training model for {instrument_name}  \n")

    model = Sequential([
        Embedding(token_size, embedding_dim, input_length=(len(moods) + len(instrument_list) + (len(instrument_list) * window_length))),
        # LSTM(256, return_sequences=True),
        # Dropout(0.15),
        LSTM(256, return_sequences=False), # Last layer set to False
        Dropout(0.15),
        Dense(token_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X, y, epochs=num_epoch, batch_size=batch_size, callbacks=[early_stopping])

    mode_path = os.path.join(model_folder, f"{instrument_name}_model.keras")
    model.save(mode_path)
    print(f'Model {instrument_name}{sample_rate}_model.keras saved to f{model_folder}')
