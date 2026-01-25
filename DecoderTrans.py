import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderTrans(nn.Module):
  def __init__(self, spanish_emb, pad_id, multi_head=4, max_len=5000):

    super().__init__()

    self.spanish_emb = spanish_emb
    self.d_model = spanish_emb.embedding_dim
    self.max_len = max_len
    self.pad_id = pad_id
    self.multi_head = multi_head
    self.d_k = self.d_model // self.multi_head
    self.vocab_size_spanish = spanish_emb.weight.shape[0]

    position = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
    even_dims = torch.arange(0, self.d_model, 2, dtype=torch.float)

    frequencies = 1.0 / (10000 ** (even_dims / self.d_model))

    pe = torch.zeros(self.max_len, self.d_model)
    pe[:, 0::2] = torch.sin(position * frequencies)
    pe[:, 1::2] = torch.cos(position * frequencies)
    pe = pe.unsqueeze(0) # (1, max_len, d_model)

    # register_buffer so it moves with .to(device) and isn't trained
    self.register_buffer("pe", pe)

    self.Wk = nn.Linear(self.d_model, self.d_model)
    self.Wq = nn.Linear(self.d_model, self.d_model)
    self.Wv = nn.Linear(self.d_model, self.d_model)
    self.Wo1 = nn.Linear(self.d_model, self.d_model)
    self.Wo2 = nn.Linear(self.d_model, self.d_model)

    self.cross_Wq = nn.Linear(self.d_model, self.d_model)

    self.enc_Wk = nn.Linear(self.d_model, self.d_model)
    self.enc_Wv = nn.Linear(self.d_model, self.d_model)

    #ForwardProp
    self.fp1 = nn.Linear(self.d_model, 2 * self.d_model)
    self.relu = nn.ReLU()
    self.fp2 = nn.Linear(2 * self.d_model, self.d_model)

    #LayerNorm
    self.ln1 = nn.LayerNorm(self.d_model)
    self.ln2 = nn.LayerNorm(self.d_model)
    self.ln3 = nn.LayerNorm(self.d_model)

    self.out = nn.Linear(self.d_model, self.vocab_size_spanish)

  def forward(self, tokens, enc_embeddings, enc_pad_mask): #tokens: (B, T), enc_pad_mask: (B, S_src)
    #Positional Encoding
    embeddings = self.spanish_emb(tokens) # (B, T, d_model)
    T = embeddings.size(1)
    device = tokens.device

    embeddings = embeddings * math.sqrt(self.d_model)
    pos_embeddings = embeddings + self.pe[:, :T, :].to(embeddings.dtype) # broadcast (1,T,d) + (B,T,d)

    #Q, K, V: (B, H, T, d_k)
    queries = self.Wq(pos_embeddings).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)
    keys = self.Wk(pos_embeddings).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)
    values = self.Wv(pos_embeddings).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)

    scores = queries @ keys.transpose(2, 3) / (math.sqrt(self.d_k)) # (B, H, T, T)

    #Causal Mask
    row_idx = torch.arange(T).unsqueeze(1).to(device)  # (T, 1)
    col_idx = torch.arange(T).unsqueeze(0).to(device)  # (1, T)

    causal_mask = col_idx > row_idx
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(causal_mask, neg)

    #Padding
    key_pad_mask = (tokens == self.pad_id) # (B, T)
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(1), neg) # mask: (B, 1, 1, T)

    #Self-Attention
    attention_weights = F.softmax(scores, dim=-1) # (B, H, T, T)
    attention_values = attention_weights @ values # (B, H, T, d_k)
    attention_values = attention_values.transpose(1, 2).reshape(pos_embeddings.size(0), pos_embeddings.size(1), self.d_model) # (B, T, d_model)

    attention_values = self.Wo1(attention_values)
    embeddings = self.ln1(attention_values + pos_embeddings)

    #dec_Q, enc_K, enc_V for enc_embeddings: (B, H, T, d_k)
    dec_queries = self.cross_Wq(embeddings).reshape(embeddings.size(0), embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)
    enc_keys = self.enc_Wk(enc_embeddings).reshape(enc_embeddings.size(0), enc_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)
    enc_values = self.enc_Wv(enc_embeddings).reshape(enc_embeddings.size(0), enc_embeddings.size(1), self.multi_head, self.d_k).transpose(1, 2)

    scores = dec_queries @ enc_keys.transpose(2, 3) / (math.sqrt(self.d_k)) # (B, H, T_dec, S_src)

    #Padding
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(enc_pad_mask.unsqueeze(1).unsqueeze(1), neg) #mask: (B, 1, 1, S_src)

    #Cross-Attention
    cross_attention_weights = F.softmax(scores, dim=-1) # (B, H, T_dec, S_src)
    cross_attention_values = cross_attention_weights @ enc_values # (B, H, T_dec, d_k)
    cross_attention_values = cross_attention_values.transpose(1, 2).reshape(embeddings.size(0), embeddings.size(1), self.d_model) # (B, T, d_model)

    cross_attention_values = self.Wo2(cross_attention_values)
    embeddings = self.ln3(cross_attention_values + embeddings) # (B, T, d_model)

    #Forward Prop
    fp = self.fp2(self.relu(self.fp1(embeddings))) # (B, T, d_model)
    embeddings = self.ln2(embeddings + fp)

    logits = self.out(embeddings) # (B, T, vocab_size)

    return logits