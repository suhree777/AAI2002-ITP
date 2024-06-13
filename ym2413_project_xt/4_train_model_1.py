import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the data with mood vectors
def load_data(data_path):
    sequences = []
    mood_vectors = []
    labels = []
    mood_labels = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}
    mood_vectors_dict = {'angry': [1, 0, 0, 0], 'happy': [0, 1, 0, 0], 'relaxed': [0, 0, 1, 0], 'sad': [0, 0, 0, 1]}
    for mood in os.listdir(data_path):
        mood_path = os.path.join(data_path, mood)
        for file in os.listdir(mood_path):
            file_path = os.path.join(mood_path, file)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                for instrument, events in data['instruments'].items():
                    if events:
                        sequences.append(events)
                        mood_vectors.append(mood_vectors_dict[mood])
                        labels.append(mood_labels[mood])
    return sequences, mood_vectors, labels

def get_max_vocab_size(vocab_directory):
    max_vocab_id = 0
    for filename in os.listdir(vocab_directory):
        filepath = os.path.join(vocab_directory, filename)
        with open(filepath, 'r') as file:
            vocab = json.load(file)
            max_id = max(map(int, vocab.values()))
            max_vocab_id = max(max_vocab_id, max_id)
    return max_vocab_id + 1

class MusicDataset(Dataset):
    def __init__(self, sequences, mood_vectors, labels):
        self.sequences = sequences
        self.mood_vectors = mood_vectors
        self.labels = labels

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.mood_vectors[idx], dtype=torch.float).to(device), 
            torch.tensor(self.sequences[idx], dtype=torch.long).to(device),
            torch.tensor(self.labels[idx], dtype=torch.long).to(device)
        )

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, num_layers=3, dropout=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(lstm_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units + 4, num_classes)  # Adjusted for mood vector

    def forward(self, mood_vector, instrument_sequence):
        x = self.embedding(instrument_sequence)
        x, _ = self.lstm(x)
        x = self.batch_norm(x[:, -1, :])
        x = torch.cat((x, mood_vector), dim=1)  # Concatenate mood vector
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model(train_loader, model, criterion, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')
    patience, trials = 5, 0  # Early stopping parameters

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        loop = tqdm(train_loader, leave=True)
        for mood_batch, sequences_batch, labels_batch in loop:
            optimizer.zero_grad()
            outputs = model(mood_batch, sequences_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item() * sequences_batch.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels_batch.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        scheduler.step(running_loss / len(train_loader.dataset))  # Adjust learning rate on plateau
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate after epoch {epoch+1}: {current_lr}")

        training_accuracy = accuracy_score(all_labels, all_predictions)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Training Accuracy: {training_accuracy:.4f}')

        # Early stopping
        val_loss = validate_model(test_loader, model, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print("Early stopping triggered")
                break

def validate_model(test_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for mood_batch, sequences_batch, labels_batch in test_loader:
            outputs = model(mood_batch, sequences_batch)
            val_loss += criterion(outputs, labels_batch).item() * labels_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    val_accuracy = correct / total
    val_loss /= len(test_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    return val_loss

if __name__ == '__main__':
    data_path = 'ym2413_project_xt/3_processed_features/data'
    sequences, mood_vectors, labels = load_data(data_path)

    sequence_lengths = [len(x) for x in sequences]
    max_seq_length = max(sequence_lengths)
    sequences_padded = [seq + [0] * (max_seq_length - len(seq)) for seq in sequences]

    vocab_directory = 'ym2413_project_xt/3_processed_features/instrument_vocabs'
    max_vocab_size = get_max_vocab_size(vocab_directory)

    X_train_seq, X_test_seq, X_train_mood, X_test_mood, y_train, y_test = train_test_split(
        sequences_padded, mood_vectors, labels, test_size=0.2, random_state=42
    )

    train_dataset = MusicDataset(X_train_seq, X_train_mood, y_train)
    test_dataset = MusicDataset(X_test_seq, X_test_mood, y_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = LSTMModel(max_vocab_size, 64, 256, 4, 0.4).to(device)  # Increase LSTM units
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower initial learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    num_epochs = 50

    train_model(train_loader, model, criterion, optimizer, scheduler, num_epochs)

    model_save_dir = 'ym2413_project_xt/4_trained_model'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'lstm_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")
