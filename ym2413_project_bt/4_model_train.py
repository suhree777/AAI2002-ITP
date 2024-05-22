import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class MusicDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        for mood_dir in os.listdir(directory):
            mood_path = os.path.join(directory, mood_dir)
            if os.path.isdir(mood_path):
                for file in os.listdir(mood_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(mood_path, file)
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            for instrument, sequence in data['instruments'].items():
                                self.data.append(torch.tensor(sequence, dtype=torch.long))
                                print(f"data print out{data}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][:-1], self.data[idx][1:]  # X and Y sequences

def split_data(dataset):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    return DataLoader(train_data, batch_size=1, shuffle=True), DataLoader(test_data, batch_size=1, shuffle=False)

# Example usage
dataset_path = 'ym2413_project_bt/processed_feature'
dataset = MusicDataset(dataset_path)
train_loader, test_loader = split_data(dataset)

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, output_size, embed_dim):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Pass input through embedding layer
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Decode the hidden state of the last time step
        return out

# Load the event vocabulary
with open('ym2413_project_bt/processed_feature/instrument_vocab.json', 'r') as file:
    event_vocab = json.load(file)
vocab_size = len(event_vocab)  # Set vocab size based on your vocabulary

# Model parameters
hidden_size = 64
num_layers = 2

# Instantiate the model
model = MusicLSTM(vocab_size, hidden_size, num_layers, vocab_size, embed_dim=50).to(device)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, epochs, train_loader, loss_function, optimizer, device):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_function(output.transpose(1, 2), y_batch)  # Adjust for expected dimensions
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        duration = time.time() - start_time  # Calculate the duration of the epoch
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Duration: {duration:.2f} seconds')

def validate_model(model, test_loader, loss_function, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output.transpose(1, 2), y_batch)
            total_loss += loss.item()
    print(f'Validation Loss: {total_loss / len(test_loader)}')

def save_model(model, model_path):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

train_model(model, 50, train_loader, loss_function, optimizer, device)
validate_model(model, test_loader, loss_function, device)

# Save the trained model
save_model(model, 'ym2413_project_bt/processed_feature/music_lstm.pth')