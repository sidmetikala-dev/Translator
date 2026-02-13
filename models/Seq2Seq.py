import math
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class Seq2Seq(L.LightningModule):
  def __init__(self, encoder, decoder, pad_en, pad_es, eos_es, bos_es, lr):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.pad_en = pad_en
    self.pad_es = pad_es
    self.eos_es = eos_es
    self.bos_es = bos_es
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_es)
    self.lr = lr

  def forward(self, batch):
    ids_en, ids_es = batch

    # Encoder
    padded_en = pad_sequence(ids_en, batch_first=True, padding_value=self.pad_en)
    padded_en = padded_en.to(self.device)
    enc_embeddings, enc_pad_mask = self.encoder(padded_en)

    # Decoder
    padded_es = pad_sequence(ids_es, batch_first=True, padding_value=self.pad_es)
    padded_es = padded_es.to(self.device)
    decoder_in = padded_es[:, :-1]
    logits = self.decoder(decoder_in, enc_embeddings, enc_pad_mask)

    # Build Targets (shift left)
    targets = padded_es[:, 1:].clone()

    return logits, targets

  def training_step(self, batch, batch_idx):

    logits, targets = self.forward(batch)

    # Compute Loss
    loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    # Accuracy (ignore PAD)
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = (targets != self.pad_es)
        correct = ((preds == targets) & mask).sum().float()
        total = mask.sum().float().clamp_min(1)
        acc = correct / total

    # Logging
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log("train_tok_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):

    logits, targets = self.forward(batch)

    # Compute Loss
    loss = self.loss_fn(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    # Accuracy
    preds = logits.argmax(dim=-1)
    mask = (targets != self.pad_es)
    acc = ((preds == targets) & mask).sum().float() / mask.sum().clamp_min(1)

    # Logging
    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("val_tok_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

  @torch.no_grad()
  def greedy_decode(self, src_ids_1d, max_len=40):
      """
      src_ids_1d: 1D LongTensor of English token IDs (includes <EOS> at end)
      Returns: 1D LongTensor of predicted Spanish IDs (starts with <BOS>, ends when <EOS> generated or max_len)
      """
      self.eval()
      device = self.device

      # Encode
      src_ids_1d = src_ids_1d.to(device)
      enc_embeddings, enc_pad_mask = self.encoder(src_ids_1d.unsqueeze(0))

      #Decode
      out = torch.tensor([[self.bos_es]], dtype=torch.long, device=device)  # (1, 1)

      for _ in range(max_len - 1):
          logits = self.decoder(out, enc_embeddings, enc_pad_mask)            # (1, t, V)
          next_id = logits[:, -1, :].argmax(dim=-1)     # (1,)

          out = torch.cat([out, next_id.unsqueeze(1)], dim=1)  # append

          # stop if EOS generated
          if next_id.item() == self.eos_es:
              break

      return out.squeeze(0)  # (t,)

  @torch.no_grad()
  def beam_search(
      self,
      src_ids_1d,
      beam_size: int = 5,
      max_len: int = 40,
      len_norm_alpha: float = 1
  ):
      self.eval()
      device = self.device

      # Encode
      src_ids_1d = src_ids_1d.to(device)
      enc_embeddings, enc_pad_mask = self.encoder(src_ids_1d.unsqueeze(0))

      # Beam items: (seq, score, ended)
      start = torch.tensor([[self.bos_es]], dtype=torch.long, device=device) #(1, 1)
      beam = [(start, 0.0, False)]
      finished = []  # store ended sequences

      def score_fn(seq, sc):
          if len_norm_alpha <= 0:
              return sc
          L = max(1, (seq.numel() - 1))
          return sc / (L ** len_norm_alpha)

      for _ in range(max_len - 1):
          candidates = []
          all_ended = True

          for seq, score, ended in beam:
              if ended:
                  candidates.append((seq, score, True))
                  continue

              all_ended = False

              logits = self.decoder(seq, enc_embeddings, enc_pad_mask)
              logprobs = F.log_softmax(logits[0, -1, :], dim=-1)  # (V,)

              topk_logp, topk_ids = torch.topk(logprobs, k=beam_size)

              for lp, tok_id in zip(topk_logp, topk_ids):
                  new_seq = torch.cat([seq, tok_id.reshape(1, 1)], dim=1)
                  new_score = score + float(lp.item())
                  new_ended = (tok_id.item() == self.eos_es)

                  item = (new_seq, new_score, new_ended)
                  candidates.append(item)

                  if new_ended:
                      finished.append(item)

          if all_ended:
              break

          # prune to top beams
          candidates.sort(key=lambda x: score_fn(x[0], x[1]), reverse=True)
          beam = candidates[:beam_size]

          # optional speed-up
          if all(b[2] for b in beam):
              break

      # Return best finished if available, else best current beam
      if finished:
          finished.sort(key=lambda x: score_fn(x[0], x[1]), reverse=True)
          best_seq = finished[0][0]
      else:
          beam.sort(key=lambda x: score_fn(x[0], x[1]), reverse=True)
          best_seq = beam[0][0]

      return best_seq

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)
