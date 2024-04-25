import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load encoded events and vocabulary
def load_data(events_path, vocab_path):
    # Load the vocabulary
    vocab = {}
    with open(vocab_path, 'r') as file:
        for line in file:
            event, index = line.strip().split('\t')
            vocab[int(index)] = event.split(',')[1].strip("() '")  # Get only the emotion part

    # Load the encoded events
    with open(events_path, 'r') as file:
        events = [int(line.strip()) for line in file.readlines()]

    # Map encoded events to their respective emotions
    labels = [vocab[event] for event in events]
    return events, labels, vocab

# Load data
events, labels, vocab = load_data('ym2413_project/encoded_events.txt', 'ym2413_project/vocabulary.txt')

# Encode labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# One-hot encode the labels
labels_categorical = to_categorical(encoded_labels, num_classes)

# Prepare dataset for training
X = np.array(events).reshape(-1, 1)  # Reshape for LSTM input
Y = labels_categorical

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

optimizer = Adam(learning_rate=0.1)

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=len(vocab), output_dim=64, input_length=1),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    LSTM(64, kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint('ym2413_project/best_chiptune_music_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model with callbacks
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])
