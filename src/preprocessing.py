import re
import string

def clean_text(text):
    # Membersihkan tag HTML
    text = re.sub(r'<br\s*/?>', ' ', text)                           
    # Membersihkan URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)               
    # Lowercase
    text = text.lower()                                              
    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()                        
    return text

def text_to_sequence(text, vocab_dict, max_len=400):
    tokens = text.split()
    # Konversi token ke ID
    unk_idx = vocab_dict.get('<UNK>', 1)
    pad_idx = vocab_dict.get('<PAD>', 0)
    
    seq = [vocab_dict.get(token, unk_idx) for token in tokens]    
    # Padding atau Truncating
    if len(seq) == 0:
        seq = [unk_idx]
        
    if len(seq) < max_len:
        seq = seq + [pad_idx] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq