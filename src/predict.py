import os
import torch
import torch.nn as nn
import pickle
from src.preprocessing import clean_text, text_to_sequence

# Definisi Arsitektur Config B (Mendukung LSTM dan GRU)
class TextRNNConfigB(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim_1=128, hidden_dim_2=64, model_type='LSTM', dropout_prob=0.6):
        super(TextRNNConfigB, self).__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if model_type == 'LSTM':
            self.rnn1 = nn.LSTM(embed_dim, hidden_dim_1, num_layers=1, batch_first=True)
            self.rnn2 = nn.LSTM(hidden_dim_1, hidden_dim_2, num_layers=1, batch_first=True)
        elif model_type == 'GRU':
            self.rnn1 = nn.GRU(embed_dim, hidden_dim_1, num_layers=1, batch_first=True)
            self.rnn2 = nn.GRU(hidden_dim_1, hidden_dim_2, num_layers=1, batch_first=True)
            
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        mask = (x != 0).unsqueeze(-1).float() 
        embeds = self.embedding(x)
        out, _ = self.rnn1(embeds)
        out, _ = self.rnn2(out)
        
        # Masked Mean Pooling
        sum_masked = (out * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_masked / lengths
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits.view(-1) 

# Fungsi Pemuatan Model Fleksibel
def load_artifacts(model_type='LSTM'):
    vocab_path = os.path.join("models", "vocab_imdb.pkl")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextRNNConfigB(vocab_size=len(vocab), embed_dim=128, hidden_dim_1=128, hidden_dim_2=64, model_type=model_type, dropout_prob=0.6)
    
    filename = "best_lstm_model.pth" if model_type == 'LSTM' else "best_gru_model.pth"
    model_path = os.path.join("models", filename)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, vocab, device

#  Prediksi
def predict_sentiment(text, model, vocab, device, max_len=400):
    cleaned_text = clean_text(text)
    seq = text_to_sequence(cleaned_text, vocab, max_len=max_len)
    
    # Batch size 1: [1, max_len]
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        # Ambil probabilitas skalar
        prob = torch.sigmoid(logits)[0].item()
        
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = (prob if prob >= 0.5 else (1.0 - prob)) * 100
    
    return sentiment, confidence