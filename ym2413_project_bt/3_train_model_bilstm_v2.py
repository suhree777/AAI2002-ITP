# Import necessary libraries
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Function to load features from JSON files
def load_features(json_folder):
    X = []
    y = []
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            file_path = os.path.join(json_folder, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)
                mood = data['mood']
                combined_features = []
                for instrument, features in data['instruments'].items():
                    # Ensure the features are in the right shape
                    instrument_features = np.array([features['pitch'], features['velocity'], features['duration'], features['tempo']])
                    combined_features.append(instrument_features.T)  # Transpose to get features in the right shape
                if combined_features:  # Check if combined_features is not empty
                    combined_features = np.concatenate(combined_features, axis=0)  # Flatten the features
                    X.append(combined_features)
                    y.append(mood)
                else:
                    print(f'No features found for {json_file}, skipping this file.')
    return X, y

# Load features
json_folder = 'ym2413_project_bt/feature_output'
X, y = load_features(json_folder)

# Pad sequences to ensure uniform length
X = pad_sequences(X, dtype='float32', padding='post', value=0.0)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert labels to numpy array
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=False)), # Output Layer return_sequences set to False
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.01)

# Compile model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save model
model.save('ym2413_project/final_chiptune_music_model.keras')