import os
import torch
import torch.nn as nn
import pickle
from src.preprocessing import clean_text, text_to_sequence

# Definisi Ulang Arsitektur Model Terbaik (LSTM - Config B)
class TextRNNConfigB(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim_1=128, hidden_dim_2=64, model_type='LSTM', dropout_prob=0.6):
        super(TextRNNConfigB, self).__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Arsitektur 2 Layer Recurrent
        if model_type == 'LSTM':
            self.rnn1 = nn.LSTM(embed_dim, hidden_dim_1, num_layers=1, batch_first=True)
            self.rnn2 = nn.LSTM(hidden_dim_1, hidden_dim_2, num_layers=1, batch_first=True)
        elif model_type == 'GRU':
            self.rnn1 = nn.GRU(embed_dim, hidden_dim_1, num_layers=1, batch_first=True)
            self.rnn2 = nn.GRU(hidden_dim_1, hidden_dim_2, num_layers=1, batch_first=True)
            
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        embeds = self.embedding(x)
        out, _ = self.rnn1(embeds)
        out, _ = self.rnn2(out)
        
        # Masked Mean Pooling
        out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(-1)

# Fungsi untuk memuat Vocab & Bobot Model
def load_artifacts():
    # Load Vocabulary
    vocab_path = os.path.join("models", "vocab_imdb.pkl")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Setup Device & Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextRNNConfigB(vocab_size=len(vocab), embed_dim=128, hidden_dim_1=128, hidden_dim_2=64, model_type='LSTM', dropout_prob=0.6)
    
    model_path = os.path.join("models", "best_lstm_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, vocab, device

# Fungsi Eksekusi Prediksi
def predict_sentiment(text, model, vocab, device, max_len=400):
    # Proses Teks (Clean & Sequence)
    cleaned_text = clean_text(text)
    seq = text_to_sequence(cleaned_text, vocab, max_len=max_len)
    
    # Ubah ke Tensor
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    
    # Inferensi
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
        
    # Konversi hasil
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = (prob if prob >= 0.5 else (1.0 - prob)) * 100
    
    return sentiment, confidence