import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader

# Check for CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset class
class MusicDataset(Dataset):
    def __init__(self, directory, vocab_size):
        self.data = []
        self.vocab_size = vocab_size
        for mood in os.listdir(directory):
            mood_path = os.path.join(directory, mood)
            for file in os.listdir(mood_path):
                with open(os.path.join(mood_path, file), 'r') as f:
                    json_data = json.load(f)
                    for instrument, events in json_data['instruments'].items():
                        self.data.append((events, mood))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        events, mood = self.data[idx]
        # One-hot encode the mood
        mood_vector = np.zeros((4,))  # Assuming 4 moods: happy, sad, angry, relaxed
        mood_index = {'happy': 0, 'sad': 1, 'angry': 2, 'relaxed': 3}[mood]
        mood_vector[mood_index] = 1
        # Pad sequences to a fixed length
        padded_events = np.pad(events, (0, 500-len(events)), mode='constant', constant_values=self.vocab_size)  # Assuming max length 500
        return torch.tensor(padded_events, dtype=torch.long), torch.tensor(mood_vector, dtype=torch.float)

# Define the LSTM model
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, mood_size):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, embedding_dim)  # +1 for padding token
        self.lstm = nn.LSTM(embedding_dim + mood_size, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, moods):
        x = self.embedding(x)
        x = torch.cat((x, moods.unsqueeze(1).repeat(1, x.size(1), 1)), 2)  # Concatenate mood to embedding
        lstm_out, _ = self.lstm(x)
        out = self.dense(lstm_out)
        return out

# Training settings
vocab_size = 50000  # Example, adjust based on your vocabulary
embedding_dim = 128
hidden_dim = 256
mood_size = 4  # Number of moods
batch_size = 128
epochs = 10

# Load data
dataset = MusicDataset('ym2413_project_xt/3_processed_features/data', vocab_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = MusicLSTM(vocab_size, embedding_dim, hidden_dim, mood_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for events, moods in data_loader:
        events, moods = events.to(device), moods.to(device)
        optimizer.zero_grad()
        outputs = model(events, moods)
        loss = criterion(outputs.transpose(1, 2), events)  # Adjust if necessary
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

# Save model
torch.save(model.state_dict(), 'music_lstm_model.pth')
