import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_instrument_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        return json.load(file)

def load_features(base_folder, vocab_path):
    instrument_vocab = load_instrument_vocab(vocab_path)
    X, y = [], []
    for mood_folder in os.listdir(base_folder):
        mood = mood_folder.split('_')[1]  # Assuming mood is indicated in the folder name
        full_path = os.path.join(base_folder, mood_folder)
        if os.path.isdir(full_path):
            for file in os.listdir(full_path):
                if file.endswith('.json'):
                    file_path = os.path.join(full_path, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        track_id = data['track_id']  # Ensure your JSON contains 'track_id'
                        track_features = {}  # Dictionary to hold features by instrument
                        for instrument, features in data['instruments'].items():
                            instrument_idx = instrument_vocab.get(instrument, -1)
                            track_features[instrument] = [
                                [instrument_idx, event['pitch'], event['velocity'], event['duration'], event['tempo']]
                                for event in features
                            ]
                        # Append a dictionary containing both track features and track_id
                        X.append({'track_id': track_id, 'features': track_features})
                        y.append(mood)
    return X, y

def scale_features(X):
    scaler = RobustScaler()
    # Example of extracting numerical data for scaling
    X_scaled = []
    for track in X:
        scaled_track_features = {}
        for instrument, features in track['features'].items():
            features_array = np.array(features, dtype=float)
            if features_array.size > 0:
                scaled_features = scaler.fit_transform(features_array)
                scaled_track_features[instrument] = scaled_features
        X_scaled.append({'track_id': track['track_id'], 'features': scaled_track_features})
    return X_scaled, scaler

# Load and prepare data
json_folder = 'ym2413_project_bt/feature_extracted'
vocab_path = 'ym2413_project_bt/feature_extracted/instrument_vocab.json'
X, y = load_features(json_folder, vocab_path)

print(f"X print out {X}")

X_scaled, scaler = scale_features(X)  # Scale features after loading them

# print(f"y print out {y}")

joblib.dump(scaler, 'scaler.pkl')

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split data
# Assuming X is a list of dictionaries containing 'track_id' and 'features'
track_ids = [track['track_id'] for track in X]
features = [track['features'] for track in X]

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, stratify=track_ids, random_state=42)
print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}")

X_train_tensors = [torch.tensor(track['features'][instrument], dtype=torch.float32) for track in X_train for instrument in track['features']]
X_test_tensors = [torch.tensor(track['features'][instrument], dtype=torch.float32) for track in X_test for instrument in track['features']]
y_train_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in y_train]
y_test_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in y_test]

X_train_padded = pad_sequence(X_train_tensors, batch_first=True, padding_value=0.0)
X_test_padded = pad_sequence(X_test_tensors, batch_first=True, padding_value=0.0)
y_train_padded = pad_sequence(y_train_tensors, batch_first=True, padding_value=0.0)
y_test_padded = pad_sequence(y_test_tensors, batch_first=True, padding_value=0.0)


# DataLoader
train_loader = DataLoader(TensorDataset(X_train_padded, y_train_padded), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_padded, y_test_padded), batch_size=16)

# Define the BiLSTM with Attention model
class MusicGeneratorBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_features):
        super(MusicGeneratorBiLSTM, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True, bidirectional=True))

        # Output layer to predict the next step in the sequence
        self.output_layer = nn.Linear(hidden_sizes[-1] * 2, num_features)

    def forward(self, x):
        for lstm in self.lstm_layers:
            lstm.flatten_parameters()
            x, _ = lstm(x)

        # Applying the output layer to each time step
        x = self.output_layer(x)
        return x

# Example: If predicting a sequence where each step has 10 possible features
num_features = 10  # This should match your feature dimension
model = MusicGeneratorBiLSTM(
    input_size=X_train.size(2),
    hidden_sizes=[64, 64, 64, 64],
    output_size=len(np.unique(y)),
    num_features=num_features
).to(device)

# print(f"input_size value is {X_train.size(2)}")
# print(f"output_size value is {len(np.unique(y))}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# Training loop with early stopping
n_epochs = 50

for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    start_time = time.time()  # Start time of the epoch

    for data, targets in train_loader:
        data = data.to(device)  # Move data to the appropriate device
        targets = targets.to(device)  # Move targets to the device

        optimizer.zero_grad()  # Clear gradients before each backward pass
        outputs = model(data)  # Forward pass: compute the predicted outputs

        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        total_loss += loss.item()  # Accumulate the loss
        
        # Assuming outputs and targets are directly comparable for accuracy:
        # E.g., outputs are logits or probabilities and targets are indices
        _, predicted = torch.max(outputs, dim=1)  # Get the predicted classes
        correct_predictions += (predicted == targets).sum().item()  # Count correct predictions
        total_predictions += targets.size(0)  # Count total predictions

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)
    accuracy = correct_predictions / total_predictions  # Calculate accuracy
    train_accuracies.append(accuracy)

    epoch_duration = time.time() - start_time  # Calculate duration

    # Optionally print the training progress
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy*100:.2f}%, Duration: {epoch_duration:.2f}s')

# Load the best model for evaluation
torch.save(model.state_dict(), 'best_model.pth')