import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

def load_data(directory, vocab_path):
    # Load instrument vocabulary
    with open(vocab_path, 'r') as vocab_file:
        instrument_vocab = json.load(vocab_file)
    
    data = []
    labels = []
    instrument_indices = {}
    mood_labels = {'Q1_happy': 0, 'Q2_angry': 1, 'Q3_sad': 2, 'Q4_relaxed': 3}

    # Map instrument names to indices
    for idx, instr in enumerate(instrument_vocab):
        instrument_indices[instr] = idx

    # Iterate through each mood directory
    for mood_dir in os.listdir(directory):
        mood_path = os.path.join(directory, mood_dir)
        if os.path.isdir(mood_path):
            mood_label = mood_labels[mood_dir]

            # Load each JSON file in the mood directory
            for file in os.listdir(mood_path):
                file_path = os.path.join(mood_path, file)
                with open(file_path, 'r') as f:
                    contents = json.load(f)
                    # Extract features and label
                    for instrument, notes in contents['instruments'].items():
                        if instrument in instrument_indices:
                            for note in notes:
                                features = [
                                    note['start_time'], 
                                    note['end_time'], 
                                    note['pitch'], 
                                    note['velocity'], 
                                    note['duration'], 
                                    note['tempo'], 
                                    instrument_indices[instrument]]
                                data.append(features)
                                labels.append(mood_label)

    return np.array(data), np.array(labels)

# Example usage
data, labels = load_data('ym2413_project_bt/feature_extracted/', 'ym2413_project_bt/feature_extracted/instrument_vocab.json')

num_features = 7  # start_time, end_time, pitch, velocity, duration, tempo, instrument_index
num_moods = 4     # happy, angry, sad, relaxed
embedding_dim = 10  # Dimension of mood embedding

# Inputs
note_input = Input(shape=(None, num_features), name='note_input')
mood_input = Input(shape=(1,), name='mood_input')

# Mood embedding
mood_embedding = Embedding(input_dim=num_moods, output_dim=embedding_dim)(mood_input)
mood_embedding = tf.squeeze(mood_embedding, axis=1)

# Use a Lambda layer to repeat the mood embedding across time steps
mood_embedding_expanded = Lambda(
    lambda x: tf.tile(tf.expand_dims(x, 1), [1, tf.shape(note_input)[1], 1]),
    output_shape=lambda input_shape: (input_shape[0], None, embedding_dim)
)(mood_embedding)
# Concatenate note input and expanded mood embedding
combined_input = Concatenate(axis=-1)([note_input, mood_embedding_expanded])

# LSTM layers
lstm_out = LSTM(64, return_sequences=True)(combined_input)
lstm_out = LSTM(64, return_sequences=True)(lstm_out)
lstm_out = LSTM(64, return_sequences=True)(lstm_out)
lstm_out = LSTM(64, return_sequences=False)(lstm_out)

# Output layers
output_pitch = Dense(128, activation='softmax')(lstm_out)  # Assuming 128 possible pitches
output_velocity = Dense(1, activation='sigmoid')(lstm_out)  # Normalized velocity

# Model
model = Model(inputs=[note_input, mood_input], outputs=[output_pitch, output_velocity])
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'])
model.summary()

# Convert labels
pitch_labels = to_categorical(labels, num_classes=128)  # Example for pitch conversion

# Splitting data into training and validation (assuming data and pitch_labels are aligned)
train_data = data[:int(len(data)*0.8)]
train_labels = pitch_labels[:int(len(data)*0.8)]
val_data = data[int(len(data)*0.8):]
val_labels = pitch_labels[int(len(data)*0.8):]

# Training
model.fit([train_data, train_labels], batch_size=32, epochs=10, validation_data=(val_data, val_labels))
model.save('path_to_my_model.keras')
