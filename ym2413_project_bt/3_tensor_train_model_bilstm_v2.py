import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Attention, Concatenate, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set GPU configuration (optional)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Function to load and normalize features from JSON files
def load_features(json_folder):
    X, y = [], []
    scaler = StandardScaler()  # Feature scaler instance
    max_length = 0  # To track the maximum sequence length
    raw_features = []  # To temporarily store feature arrays before padding

    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            with open(os.path.join(json_folder, json_file), 'r') as file:
                data = json.load(file)
                mood = data['mood']
                all_features = []
                for instrument, features in data['instruments'].items():
                    features_array = np.array([features['pitch'], features['velocity'], features['duration'], features['tempo']]).T
                    scaled_features = scaler.fit_transform(features_array)
                    all_features.append(scaled_features)
                if all_features:
                    combined_features = np.concatenate(all_features, axis=0)
                    raw_features.append(combined_features)
                    y.append(mood)
                    if combined_features.shape[0] > max_length:
                        max_length = combined_features.shape[0]

    # Pad sequences after finding max length
    X = pad_sequences(raw_features, maxlen=max_length, padding='post', dtype='float32', value=0.0)
    return np.array(X), np.array(y)

# Load and prepare data
json_folder = 'ym2413_project_bt/feature_output'
X, y = load_features(json_folder)
y = LabelEncoder().fit_transform(y)

# Pad sequences for uniform input size
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post", dtype='float32')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the BiLSTM with Attention model
input_layer = Input(shape=(None, X_train.shape[2]))
lstm = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
attention = Attention()([lstm, lstm])
context_vector = Concatenate()([lstm, attention])
# Apply global average pooling to reduce the sequence dimension
pooled_vector = GlobalAveragePooling1D()(context_vector)
dropout = Dropout(0.2)(pooled_vector)
output_layer = Dense(len(np.unique(y)), activation='softmax')(dropout)
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training loop
history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}%")

# Optionally save the model
model.save('bilstm_model.h5')
