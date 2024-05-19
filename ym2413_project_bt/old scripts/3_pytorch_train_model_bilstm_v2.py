import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Function to load and normalize features from JSON files
def load_features(json_folder):
    X, y = [], []
    scaler = StandardScaler()  # Feature scaler instance
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            with open(os.path.join(json_folder, json_file), 'r') as file:
                data = json.load(file)
                mood = data['mood']
                all_features = []
                for instrument, features in data['instruments'].items():
                    # Combine and scale features
                    features_array = np.array([features['pitch'], features['velocity'], features['duration'], features['tempo']]).T
                    scaled_features = scaler.fit_transform(features_array)
                    all_features.append(scaled_features)
                if all_features:
                    combined_features = np.concatenate(all_features, axis=0)
                    X.append(combined_features)
                    y.append(mood)
                else:
                    print(f"No features found in {json_file}, skipping.")
    return X, y

# Load and prepare data
json_folder = 'ym2413_project_bt/feature_output'
X, y = load_features(json_folder)
X = [torch.tensor(x, dtype=torch.float32) for x in X]
X = pad_sequence(X, batch_first=True, padding_value=0.0)
y = LabelEncoder().fit_transform(y)
y = torch.tensor(y, dtype=torch.long)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}")

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# Define the BiLSTM with Attention model
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(AttentionBiLSTM, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.layers = nn.ModuleList()
        
        # Creating each layer with specified hidden sizes
        for i in range(self.num_layers):
            # Adjust input size for subsequent layers
            input_dim = input_size if i == 0 else hidden_sizes[i-1] * 2
            self.layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True, bidirectional=True))

        # Attention mechanism: Assuming the last LSTM layer's size to determine the attention layer size
        last_hidden_size = hidden_sizes[-1] * 2
        self.attention_layer = nn.Sequential(
            nn.Linear(last_hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Final fully connected layer
        self.fc = nn.Linear(last_hidden_size, num_classes)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output, _ = layer(output)
        
        # Apply attention
        attention_weights = F.softmax(self.attention_layer(output), dim=1)
        context_vector = torch.sum(attention_weights * output, dim=1)
        
        # Final classification layer
        out = self.fc(context_vector)
        return out

# Hyperparameters and model setup
model = AttentionBiLSTM(
    input_size=X_train.size(2), 
    hidden_sizes=[128, 128, 128], 
    num_classes=len(np.unique(y))
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with early stopping
best_accuracy = 0
epochs_no_improve = 0
n_epochs_stop = 20

for epoch in range(20):
    model.train()
    start_time = time.time()  # Start time for the epoch
    running_loss, correct, total = 0.0, 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    epoch_duration = time.time() - start_time
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%, Duration: {epoch_duration:.2f}s')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'bilstm_model.pth')  # Save the best model
    else:
        epochs_no_improve += 1
    if epochs_no_improve == n_epochs_stop:
        print('Early stopping triggered.')
        break

# Load the best model for evaluation
model.load_state_dict(torch.load('bilstm_model.pth'))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
