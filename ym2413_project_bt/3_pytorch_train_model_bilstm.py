import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
print(X)
# Pad sequences to ensure uniform length
X = [torch.tensor(x, dtype=torch.float32) for x in X]
print(X)
X = pad_sequence(X, batch_first=True, padding_value=0.0)
print(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = torch.tensor(y, dtype=torch.long)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = X_train.size(2)
hidden_size = 64
num_layers = 3
num_classes = len(label_encoder.classes_)
learning_rate = 0.001
num_epochs = 20

# Initialize model, loss function, and optimizer
model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    start_time = time.time()  # Start time for the epoch
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    end_time = time.time()  # End time for the epoch
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {end_time - start_time:.2f}s')


# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'bilstm_model.pth')