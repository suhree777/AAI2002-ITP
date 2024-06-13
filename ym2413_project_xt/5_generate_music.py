import torch
import json
import os
import torch.nn as nn

# Load the trained model
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

def load_model(model_path, vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate):
    model = LSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_max_vocab_size(vocab_directory):
    max_vocab_id = 0
    for filename in os.listdir(vocab_directory):
        filepath = os.path.join(vocab_directory, filename)
        with open(filepath, 'r') as file:
            vocab = json.load(file)
            max_id = max(map(int, vocab.values()))
            max_vocab_id = max(max_vocab_id, max_id)
    return max_vocab_id + 1

# Map mood input to mood vector
def get_mood_vector(mood):
    mood_vectors_dict = {'angry': [1, 0, 0, 0], 'happy': [0, 1, 0, 0], 'relaxed': [0, 0, 1, 0], 'sad': [0, 0, 0, 1]}
    return torch.tensor(mood_vectors_dict[mood], dtype=torch.float)

# Function to generate music
def generate_music(model, mood_vector, max_sequence_length):
    # Initialize with a sequence of zeros
    current_sequence = torch.zeros(1, 1).long().to(device)
    generated_sequence = []

    with torch.no_grad():
        while len(generated_sequence) < max_sequence_length:
            outputs = model(mood_vector, current_sequence)
            predicted = torch.argmax(outputs, dim=1)
            generated_sequence.append(predicted.item())
            current_sequence = torch.cat([current_sequence, predicted.unsqueeze(0)], dim=1)

    return generated_sequence

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths and parameters
    model_path = 'ym2413_project_xt/4_trained_model/lstm_model.pth'
    vocab_size = get_max_vocab_size('ym2413_project_xt/3_processed_features/instrument_vocabs')
    embedding_dim = 64
    lstm_units = 256
    num_classes = 4  # Assuming 4 mood classes
    dropout_rate = 0.4

    # Load the trained model
    model = load_model(model_path, vocab_size, embedding_dim, lstm_units, num_classes, dropout_rate).to(device)

    # Get user input for mood
    user_mood = input("Enter the mood (angry, happy, sad, relaxed): ").lower()

    if user_mood not in ['angry', 'happy', 'sad', 'relaxed']:
        print("Invalid mood input. Please choose from: angry, happy, sad, relaxed.")
    else:
        # Convert mood to mood vector
        mood_vector = get_mood_vector(user_mood).unsqueeze(0).to(device)

        # Generate music based on the mood
        max_sequence_length = 100  # Adjust as needed based on your maximum sequence length
        generated_sequence = generate_music(model, mood_vector, max_sequence_length)

        print(f"Generated sequence based on '{user_mood}' mood:")
        print(generated_sequence)
