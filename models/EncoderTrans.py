import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderTrans(nn.Module):
  def __init__(self, english_emb, pad_id, multi_head=4, max_len=5000):

    super().__init__()

    self.english_emb = english_emb
    self.d_model = english_emb.embedding_dim
    self.max_len = max_len
    self.pad_id = pad_id
    self.multi_head = multi_head
    self.d_k = self.d_model // self.multi_head

    position = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
    even_dims = torch.arange(0, self.d_model, 2, dtype=torch.float)

    frequencies = 1.0 / (10000 ** (even_dims / self.d_model))

    pe = torch.zeros(self.max_len, self.d_model)
    pe[:, 0::2] = torch.sin(position * frequencies)
    pe[:, 1::2] = torch.cos(position * frequencies)
    pe = pe.unsqueeze(0) # (1, max_len, d_model)

    # register_buffer so it moves with .to(device) and isn't trained
    self.register_buffer("pe", pe)

    self.Wq = nn.Linear(self.d_model, self.d_model)
    self.Wk = nn.Linear(self.d_model, self.d_model)
    self.Wv = nn.Linear(self.d_model, self.d_model)
    self.Wo = nn.Linear(self.d_model, self.d_model)

    #ForwardProp
    self.fp1 = nn.Linear(self.d_model, 2 * self.d_model)
    self.relu = nn.ReLU()
    self.fp2 = nn.Linear(2 * self.d_model, self.d_model)

    #LayerNorm
    self.ln1 = nn.LayerNorm(self.d_model)
    self.ln2 = nn.LayerNorm(self.d_model)

  def forward(self, tokens): # tokens: (B, T)
    #Positional encoding
    embeddings = self.english_emb(tokens) # (B, T, d_model)
    T = embeddings.size(1)

    embeddings = embeddings * math.sqrt(self.d_model)
    pos_embeddings = embeddings + self.pe[:, :T, :].to(embeddings.dtype) # broadcast (1,T,d) + (B,T,d)

    #Q, K, V: (B, H, T, d_k)
    queries = self.Wq(pos_embeddings).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)
    keys = self.Wk(pos_embeddings).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)
    values = self.Wv(pos_embeddings).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)

    scores = queries @ keys.transpose(2, 3) / (math.sqrt(self.d_k)) # (B, H, T, T)

    #Padding
    key_pad_mask = (tokens == self.pad_id) # (B, T)
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(1), neg) # (B, 1, 1, T)

    #Self-Attention
    attention_weights = F.softmax(scores, dim=-1) # (B, H, T, T)
    attention_values = attention_weights @ values # (B, H, T, d_k)
    attention_values = attention_values.transpose(1, 2).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.d_model) # (B, T, d_model)

    attention_values = self.Wo(attention_values)
    embeddings = self.ln1(attention_values + pos_embeddings)

    #Forward Prop
    fp = self.fp2(self.relu(self.fp1(embeddings))) # (B, T, d_model)

    embeddings = self.ln2(embeddings + fp)

    return embeddings, key_pad_mask