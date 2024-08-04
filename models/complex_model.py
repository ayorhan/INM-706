# complex_model.py
import torch
import torch.nn as nn

class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout):
        super(BiLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attn_output = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(self.dropout(attn_output))
        return output

def init_model(vocab_size):
    embed_dim = 128
    hidden_dim = 256
    output_dim = 1
    num_layers = 2
    dropout = 0.5
    model = BiLSTMWithAttention(vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout)
    return model
