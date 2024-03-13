import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
num_layers = 2
num_epochs = 5
batch_size = 64
seq_length = 30
learning_rate = 0.002

# Define dataset path
dataset_path = './data/penn'

# Load and tokenize dataset
print("Tokenizing dataset...")
tokenizer = get_tokenizer('basic_english')
def read_penn_treebank(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield tokenizer(line)

train_iter = read_penn_treebank(os.path.join(dataset_path, 'train.txt'))
val_iter = read_penn_treebank(os.path.join(dataset_path, 'valid.txt'))
test_iter = read_penn_treebank(os.path.join(dataset_path, 'test.txt'))

vocab = build_vocab_from_iterator(train_iter, specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Reset iterators for data processing
print("Building vocabulary...")
train_iter = read_penn_treebank(os.path.join(dataset_path, 'train.txt'))
val_iter = read_penn_treebank(os.path.join(dataset_path, 'valid.txt'))
test_iter = read_penn_treebank(os.path.join(dataset_path, 'test.txt'))

# Data processing functions
def data_process(raw_text_iter):
    data = [torch.tensor([vocab[token] for token in item], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

print("Processing training data...")
train_data = data_process(train_iter)
print("Processing validation data...")
val_data = data_process(val_iter)
print("Processing test data...")
test_data = data_process(test_iter)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, batch_size)
test_data = batchify(test_data, batch_size)

# Define the LSTM model
print("Defining LSTM model...")
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Initialize the model, loss function, and optimizer
model = LSTMModel(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to generate input and target sequences
def get_batch(source, i):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_length)):
        if batch % 100 == 0:
            print(f"Processing batch {batch}/{len(train_data)//seq_length}")

        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, len(vocab)), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    elapsed_time = time.time() - start_time
    print(f'Epoch {epoch+1}, Loss: {total_loss / (batch+1)}, Time: {elapsed_time:.2f} seconds')


# Save the pre-trained model
print("Saving pre-trained model...")
torch.save(model.state_dict(), 'model2/pretrained_lstm_model.pth')
print("Pre-trained model saved.")
