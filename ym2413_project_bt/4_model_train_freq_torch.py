import os
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

for i in range(20):
    print("\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def mood_to_onehot(mood, moods):
    one_hot = torch.zeros(len(moods)).to(device)
    index = moods.index(mood)
    one_hot[index] = 1
    return one_hot

def load_and_normalize_all_instruments(filepath, instrument_list):
    with open(filepath, 'r') as file:
        data = json.load(file)

    mood_vector = mood_to_onehot(data['mood'], moods)
    instrument_vector = torch.tensor(data['instrument_vector'], dtype=torch.float32).to(device)
    feature_vector = torch.cat([mood_vector, instrument_vector])

    all_instrument_data = []
    max_length = 0

    instrument_data_arrays = {}
    for instrument in instrument_list:
        if instrument in data['instruments']:
            instrument_data = torch.tensor(data['instruments'][instrument], dtype=torch.float32).to(device)
            max_length = max(max_length, instrument_data.size(0))
            instrument_data_arrays[instrument] = instrument_data
        else:
            instrument_data_arrays[instrument] = torch.tensor([], dtype=torch.float32).to(device)

    for instrument in instrument_list:
        instrument_data = instrument_data_arrays[instrument]
        if instrument_data.nelement() > 0:
            padded_data = torch.nn.functional.pad(instrument_data, (0, max_length - instrument_data.size(0)))
        else:
            padded_data = torch.zeros(max_length, device=device)
        all_instrument_data.append(padded_data)

    combined_instrument_data = torch.stack(all_instrument_data, dim=-1)
    return feature_vector, combined_instrument_data

def create_sliding_window_dataset(feature_vector, instrument_data, window_length, target_instrument_index):
    X = []
    y = []
    num_instruments = instrument_data.size(1)
    sequence_length = instrument_data.size(0)

    initial_feature_vector = torch.cat([feature_vector] + [torch.zeros(window_length, device=device) for _ in range(num_instruments)])
    initial_targets = instrument_data[0, target_instrument_index]
    X.append(initial_feature_vector)
    y.append(initial_targets)

    for start in range(sequence_length - window_length):
        end = start + window_length
        windows = [instrument_data[start:end, i] for i in range(num_instruments)]
        full_sequence = torch.cat([feature_vector] + windows)
        if end < sequence_length:
            target = instrument_data[end, target_instrument_index]
            X.append(full_sequence)
            y.append(target)

    return torch.stack(X), torch.tensor(y, device=device)

def process_dataset(root_dir, instrument_list, window_length, index):
    all_X = []
    all_y = []

    for file in os.listdir(root_dir):
        filepath = os.path.join(root_dir, file)
        feature_vector, instrument_data = load_and_normalize_all_instruments(filepath, instrument_list)
        X, y = create_sliding_window_dataset(feature_vector, instrument_data, window_length, index)
        all_X.append(X)
        all_y.append(y)
    print("All eligible data has been processed")
    return len(feature_vector), torch.cat(all_X), torch.cat(all_y)

class MusicModel(nn.Module):
    def __init__(self, token_size, embedding_dim):
        super(MusicModel, self).__init__()
        self.embedding = nn.Embedding(token_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.15)
        # self.lstm2 = nn.LSTM(256, 256, batch_first=True)
        # self.dropout2 = nn.Dropout(0.15)
        self.dense = nn.Linear(256, token_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)  # x has shape [batch, sequence_length, 256]
        x = self.dropout1(x)
        # x, _ = self.lstm2(x)  # x has shape [batch, sequence_length, 256]
        # x = self.dropout2(x)
        x = self.dense(x[:, -1, :])  # Selecting the last timestep output for each sequence
        return x

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float().sum()
    return correct / y_true.shape[0]

def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}')
        epoch_start_time = time.time()
        total_loss = 0
        total_accuracy = 0
        model.train()

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.long().to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            accuracy = calculate_accuracy(outputs, batch_y)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            loss.backward()
            optimizer.step()

        epoch_duration = time.time() - epoch_start_time
        average_loss = total_loss / len(dataloader)
        average_accuracy = total_accuracy / len(dataloader)
        print(f'Epoch {epoch+1}: Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}, Duration: {epoch_duration:.2f} seconds')

    return model


directory_path = 'ym2413_project_bt/3_processed_freq/'
model_folder = 'ym2413_project_bt/model_folder_freq'
os.makedirs(model_folder, exist_ok=True)
summary_path = os.path.join(directory_path, 'summary.json')
with open(summary_path, 'r') as f:
    data = json.load(f)
    duration_range = data['duration_range']
    instrument_list = data['top_instrument_names']
    moods = data['mood_labels']
    sample_rate = data['desired_sample_rate']
    processed_count = data['processed_count']

print(f"Summary file list has been loaded")
print(f"Duration range: {duration_range}")
print(f"Desired sample rate: {sample_rate}")
print(f"Mood labels: {moods}")
print(f"Instrument list of {len(instrument_list)}: {instrument_list}")
print(f"Total files processed: {processed_count}")

vocab_file_path = os.path.join(directory_path, 'event_vocab.json')
with open(vocab_file_path, 'r') as f:
    vocab = json.load(f)

token_size = len(vocab) + len(instrument_list) + len(moods)
size_multiplyer = 5
window_length = sample_rate * size_multiplyer # Sliding window size per instrment
embedding_dim = 512
num_epoch = 3
batch_size = 256

data['size_multiplyer'] = size_multiplyer
data['embedding_dim'] = embedding_dim

# Write the updated data back to the same file
with open(summary_path, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Sliding Window length: {window_length}")

for index, instrument_name in enumerate(instrument_list):
    num_features, X, y = process_dataset(os.path.join(directory_path, 'data/'), instrument_list, window_length, index)
    model = MusicModel(token_size, embedding_dim).to(device)
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = train_model(model, dataloader, optimizer, criterion, num_epoch, device)
    """    
    for epoch in range(num_epoch):
        print(f'Starting Epoch {epoch+1}')
        epoch_start_time = time.time()
        total_loss = 0
        total_accuracy = 0
        model.train()  # Set model to training mode

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.long().to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            accuracy = calculate_accuracy(outputs, batch_y)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            loss.backward()
            optimizer.step()

        epoch_duration = time.time() - epoch_start_time
        average_loss = total_loss / len(dataloader)
        average_accuracy = total_accuracy / len(dataloader)
        
        print(f'Epoch {epoch+1}: Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}, Duration: {epoch_duration:.2f} seconds')
    """
    model_path = os.path.join(model_folder, f"{instrument_name}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f'Model for {instrument_name} saved to {model_folder}')
