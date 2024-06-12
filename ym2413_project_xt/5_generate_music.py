import torch
import torch.nn as nn
import torch.nn.functional as F
import pretty_midi
import numpy as np
import os
import json
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, num_layers=3, dropout=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(lstm_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.batch_norm(x[:, -1, :])
        x = self.dropout(x)
        x = self.fc(x)
        return x

def load_model(model_path, vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate):
    model = LSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def load_vocab(file_path):
    with open(file_path, 'r') as file:
        vocab = json.load(file)
    return {int(k): ast.literal_eval(v) for k, v in vocab.items() if k.isdigit()}

def main():
    model_path = 'ym2413_project_xt/4_trained_model/trained_lstm_model.pth'
    vocab_path = 'ym2413_project_xt/3_processed_features/instrument_vocabs/'

    # List available instruments
    instruments = [file.split('_')[0] for file in os.listdir(vocab_path) if file.endswith('_vocab.json')]
    for idx, instr in enumerate(instruments):
        print(f"{idx + 1}. {instr}")

    # User selects the instrument
    choice = int(input("Select an instrument by number: "))
    instrument = instruments[choice - 1]
    vocab_file = os.path.join(vocab_path, f'{instrument}_vocab.json')
    
    vocab = load_vocab(vocab_file)
    vocab_size = len(vocab) + 1  # Plus one for padding index

    embedding_dim = 64
    lstm_units = 256  # Must match the original training configuration
    dropout_rate = 0.4  # Must match the original training configuration
    model = load_model(model_path, vocab_size, embedding_dim, lstm_units, vocab_size, dropout_rate)

    # Example seed sequence (adjust as needed)
    seed_length = 50
    current_seq = torch.randint(0, vocab_size, (1, seed_length)).to(device)
    
    # Assume some generation length
    generated_sequence = generate_music(model, current_seq, 300)  # Adjust the generation length and temperature if needed
    print("Generated sequence:", generated_sequence)

def generate_music(model, current_seq, generation_length, temperature=1.0):
    model.eval()
    generated_seq = current_seq.tolist()[0]
    with torch.no_grad():
        for _ in range(generation_length):
            output = model(current_seq)
            probabilities = F.softmax(output / temperature, dim=-1).squeeze()
            next_note = torch.multinomial(probabilities[-1], 1)
            generated_seq.append(next_note.item())
            current_seq = torch.cat((current_seq[:, 1:], next_note.unsqueeze(0)), dim=1)
    return generated_seq

if __name__ == '__main__':
    main()
