import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
        full_path = os.path.join(base_folder, mood_folder)
        if os.path.isdir(full_path):
            mood = mood_folder.split('_')[1]  # Assuming folder name format 'Q1_happy', etc.
            for file in os.listdir(full_path):
                if file.endswith('.json'):
                    file_path = os.path.join(full_path, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        all_features = []
                        for instrument, features in data['instruments'].items():
                            instrument_idx = instrument_vocab.get(instrument, -1)  # Default to -1 if not found
                            instrument_features = []
                            for event in features:
                                instrument_features.append([
                                    instrument_idx,  # Add instrument index as a feature
                                    event['pitch'],
                                    event['velocity'],
                                    event['duration'],
                                    event['tempo']
                                ])
                            if instrument_features:
                                instrument_features_array = np.array(instrument_features)
                                all_features.append(instrument_features_array)
                        if all_features:
                            combined_features = np.concatenate(all_features, axis=0)
                            X.append(combined_features)
                            y.append(mood)
                        else:
                            print(f"No features found in {file}, skipping.")
    return X, y

def scale_features(X):
    scaler = StandardScaler()  # Feature scaler instance
    # Flatten X to fit the scaler, then transform each element
    all_data = np.concatenate(X, axis=0)
    scaler.fit(all_data)
    # Transform each array of features
    scaled_X = [scaler.transform(x) for x in X]
    return scaled_X, scaler  

def verify_data_integrity(X):
    for i, x in enumerate(X):
        if np.isnan(x).any() or np.isinf(x).any():
            print(f"Data integrity issue found in batch {i}")

# Load and prepare data
json_folder = 'ym2413_project_bt/feature_extracted'
vocab_path = 'ym2413_project_bt/feature_extracted/instrument_vocab.json'
X, y = load_features(json_folder, vocab_path)
# print(f"X print out {X}")
# print(f"y print out {X}")
X, scaler = scale_features(X)  # Scale features after loading them
verify_data_integrity(X)
X = [torch.tensor(x, dtype=torch.float32) for x in X]
X = pad_sequence(X, batch_first=True, padding_value=0.0)
y = LabelEncoder().fit_transform(y)
y = torch.tensor(y, dtype=torch.long)

# Optionally save the scaler for later use in generation/inference
joblib.dump(scaler, 'scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}")

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# Define the BiLSTM with Attention model
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(AttentionBiLSTM, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True, bidirectional=True))
        
        # Attention layer (optional)
        self.attention = nn.Linear(hidden_sizes[-1] * 2, 1)  # Attention across bidirectional output
        
        # Output layer to predict music features
        self.output_layer = nn.Linear(hidden_sizes[-1] * 2, output_size)  # Adjust output size based on the number of features

    def forward(self, x):
        # Passing input through LSTM layers
        for lstm in self.lstm_layers:
            lstm.flatten_parameters()
            x, _ = lstm(x)
        
        # Applying attention (if needed)
        attention_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(attention_weights * x, dim=1, keepdim=True)
        
        # Generating output for each timestep
        x = self.output_layer(x)
        return x

# Hyperparameters and model setup
model = AttentionBiLSTM(
    input_size=X_train.size(2), 
    hidden_sizes=[64, 64, 64, 64], 
    output_size=len(np.unique(y))
).to(device)

# print(f"input_size value is {X_train.size(2)}")
# print(f"output_size value is {len(np.unique(y))}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# Training loop with early stopping
best_accuracy = 0
epochs_no_improve = 0
epoch_cycle = 360
n_epochs_stop = 40



for epoch in range(epoch_cycle):
    model.train()
    start_time = time.time()
    running_loss, correct, total = 0.0, 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        print("Input shape:", data.shape)
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_duration = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Duration: {epoch_duration:.2f}s')

    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'bilstm_model.pth')
    else:
        epochs_no_improve += 1

    if epochs_no_improve == n_epochs_stop:
        print('Early stopping triggered.')
        break

# Load the best model for evaluation
model.load_state_dict(torch.load('bilstm_model.pth'))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')

# Plotting training and validation loss and accuracy
plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()