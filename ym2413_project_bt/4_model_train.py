import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torch.optim import Adam
from transformers import TransfoXLModel, TransfoXLConfig

def mood_to_onehot(mood):
    # This function converts a mood string to a one-hot encoded vector
    moods = ['happy', 'angry', 'sad', 'relaxed']  # Adjust as per your moods
    one_hot = np.zeros(len(moods))
    index = moods.index(mood)
    one_hot[index] = 1
    return one_hot

def load_and_normalize_data(filepath, instrument_name, sequence_length=50):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    if instrument_name not in data['instruments']:
        return None  # Skip files where the instrument is not present

    # Extract mood and instrument_vector
    mood_vector = mood_to_onehot(data['mood'])
    instrument_vector = data['instrument_vector']
    feature_vector = np.concatenate([mood_vector, instrument_vector])
    
    # Extract instrument data
    instrument_data = np.array(data['instruments'][instrument_name])
    if len(instrument_data) > sequence_length:
        instrument_data = instrument_data[:sequence_length]  # Truncate
    elif len(instrument_data) < sequence_length:
        instrument_data = np.pad(instrument_data, (0, sequence_length - len(instrument_data)), 'constant')
    
    final_input = np.concatenate([feature_vector, instrument_data])
    # print(f"Processed data for {instrument_name}: {instrument_data}")
    return final_input

def process_instrument_data(root_dir, instrument_name, sequence_length=50):
    all_data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(subdir, file)
                try:
                    processed_data = load_and_normalize_data(filepath, instrument_name, sequence_length)
                    all_data.append(processed_data)
                except Exception as e:
                    print(f"Failed to process {filepath}: {str(e)}")
    return all_data

for i in range(20):
    print("\n")

instrument_list = ["Acoustic Bass","Acoustic Grand Piano","Acoustic Guitar (nylon)","Church Organ",
                   "Clarinet","Electric Guitar (clean)","Electric Piano 1","Flute","French Horn",
                   "Harpsichord","Lead 2 (sawtooth)","Oboe","Synth Bass 1","Trumpet","Vibraphone",
                   "Violin"]

directory_path = 'ym2413_project_bt/3_processed_feature_limited/data/'
instrument_name = 'Acoustic Bass'  # Specified the instrument

instrument_datasets = process_instrument_data(directory_path, instrument_name)
# print(f"instrument_datasets data for {instrument_datasets}")
print(f"instrument_datasets data: {len(instrument_datasets)} samples")

def clean_data(data):
    return [x for x in data if x is not None]

instrument_datasets = clean_data(instrument_datasets)
print(f"Cleaned instrument_datasets data: {len(instrument_datasets)} samples")

instrument_datasets = np.array(instrument_datasets)
np.random.shuffle(instrument_datasets)

train_data, test_data = train_test_split(instrument_datasets, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=42)

train_inputs, train_labels = torch.Tensor(train_data[:, :-1]), torch.Tensor(train_data[:, 1:])
val_inputs, val_labels = torch.Tensor(val_data[:, :-1]), torch.Tensor(val_data[:, 1:])
test_inputs, test_labels = torch.Tensor(test_data[:, :-1]), torch.Tensor(test_data[:, 1:])

train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)
test_dataset = TensorDataset(test_inputs, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data loaders created!")
vocab_file_path = os.path.join('ym2413_project_bt/3_processed_feature_limited/instrument_vocabs/', f"{instrument_name}_vocab.json")

with open(vocab_file_path, 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)  # The total number of unique tokens

print(f"Loaded vocabulary for {instrument_name} with size: {vocab_size}")

# Model configuration
config = TransfoXLConfig(
    vocab_size=vocab_size,
    d_model=512,       # Dimension of the model's embeddings
    n_head=8,          # Number of attention heads
    num_layers=6,      # Number of transformer layers
    dropout=0.1,       # Dropout rate
    adaptive=True,
    cutoffs=[2000, 6000],
    div_val=4
)

model = TransfoXLModel(config)
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model configured with vocab size: {vocab_size}")
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(model, data_loader, optimizer, loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, dtype=torch.long), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Assuming `train_loader` is defined as above
train(model, train_loader, optimizer, loss_fn, epochs=10)

