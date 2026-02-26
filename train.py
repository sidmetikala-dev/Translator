"""
train.py — Training pipeline for English → Spanish Transformer.

This script:
- Downloads and preprocesses sentence-pair data
- Builds vocabularies and token mappings
- Initializes a custom Transformer encoder–decoder
- Trains using teacher forcing with padding-aware cross-entropy loss
- Saves model checkpoints based on validation loss
- Supports resume-from-checkpoint training

This file exists to make the training and checkpointing logic explicit
and reproducible outside of a Jupyter notebook environment.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import kagglehub
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from models.Seq2Seq import Seq2Seq
from models.EncoderTrans import EncoderTrans
from models.DecoderTrans import DecoderTrans


# -----------------------------
# Tokenization
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:'[A-Za-z]+)?|[.,!?;:]", flags=0)

def fast_tok(s: str):
    return TOKEN_RE.findall(str(s))


# -----------------------------
# Dataset
# -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, src_ids, trg_ids):
        self.src = src_ids
        self.trg = trg_ids

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]


def collate_fn(batch):
    ids_en = [item[0] for item in batch]
    ids_es = [item[1] for item in batch]
    return ids_en, ids_es


def main():
    # -----------------------------
    # 0) Download / load data
    # -----------------------------
    # NOTE: keep this even if you already have weights. It documents how you trained.
    path = kagglehub.dataset_download("lonnieqin/englishspanish-translation-dataset")
    # Your notebook used "data.csv" — keep same. If KaggleHub changes file name, adjust here.
    csv_path = os.path.join(path, "data.csv")
    data = pd.read_csv(csv_path)

    # Expect columns: 'english', 'spanish'
    if "english" not in data.columns or "spanish" not in data.columns:
        raise ValueError("CSV must contain 'english' and 'spanish' columns.")

    # -----------------------------
    # 1) Tokenize
    # -----------------------------
    english_tokens = [fast_tok(s) for s in data["english"]]
    spanish_tokens = [fast_tok(s) for s in data["spanish"]]

    # -----------------------------
    # 2) Build English vocab
    # -----------------------------
    index_dict_english = {}
    for sen in english_tokens:
        for word in sen:
            if word not in index_dict_english:
                index_dict_english[word] = len(index_dict_english)

    index_dict_english["<EOS>"] = len(index_dict_english)
    PAD_EN = len(index_dict_english)
    index_dict_english["<PAD>"] = PAD_EN

    index_dict_english["<UNK>"] = len(index_dict_english)
    UNK_EN = index_dict_english["<UNK>"]

    # -----------------------------
    # 3) Build Spanish vocab
    # -----------------------------
    index_dict_spanish = {}
    for sen in spanish_tokens:
        for word in sen:
            if word not in index_dict_spanish:
                index_dict_spanish[word] = len(index_dict_spanish)

    index_dict_spanish["<EOS>"] = len(index_dict_spanish)
    index_dict_spanish["<BOS>"] = len(index_dict_spanish)
    PAD_ES = len(index_dict_spanish)
    index_dict_spanish["<PAD>"] = PAD_ES

    # -----------------------------
    # 4) Embeddings
    # -----------------------------
    vocab_size_english = len(index_dict_english)
    vocab_size_spanish = len(index_dict_spanish)
    embedding_dim = 256

    english_emb = nn.Embedding(vocab_size_english, embedding_dim, padding_idx=PAD_EN)
    spanish_emb = nn.Embedding(vocab_size_spanish, embedding_dim, padding_idx=PAD_ES)

    # -----------------------------
    # 5) Convert sentences -> id tensors
    # -----------------------------
    english_ids = []
    spanish_ids = []

    for sen in english_tokens:
        ids = [index_dict_english.get(word, UNK_EN) for word in sen]
        ids.append(index_dict_english["<EOS>"])
        english_ids.append(torch.tensor(ids, dtype=torch.long))

    for sen in spanish_tokens:
        ids = (
            [index_dict_spanish["<BOS>"]]
            + [index_dict_spanish[word] for word in sen]
            + [index_dict_spanish["<EOS>"]]
        )
        spanish_ids.append(torch.tensor(ids, dtype=torch.long))

    # -----------------------------
    # 6) Shuffle
    # -----------------------------
    indices = np.random.permutation(len(english_ids))
    shuffled_english_ids = [english_ids[i] for i in indices]
    shuffled_spanish_ids = [spanish_ids[i] for i in indices]

    # -----------------------------
    # 7) Split (80/10/10)
    # -----------------------------
    size = data.shape[0]  # ~118964 in your notebook
    train_size = int(size * 0.8)
    test_size = int(size * 0.1)
    val_size = size - train_size - test_size

    train_data = TranslationDataset(
        shuffled_english_ids[:train_size],
        shuffled_spanish_ids[:train_size]
    )

    # test_data exists in notebook, but train.py only needs train/val
    val_data = TranslationDataset(
        shuffled_english_ids[-val_size:],
        shuffled_spanish_ids[-val_size:]
    )

    # -----------------------------
    # 8) DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # -----------------------------
    # 9) Lightning model + checkpoint saving
    # -----------------------------
    model = Seq2Seq(
        encoder=EncoderTrans(english_emb, PAD_EN),
        decoder=DecoderTrans(spanish_emb, PAD_ES),
        pad_en=PAD_EN,
        pad_es=PAD_ES,
        eos_es=index_dict_spanish["<EOS>"],
        bos_es=index_dict_spanish["<BOS>"],
        lr=3e-4
    )

    logger = TensorBoardLogger(save_dir="logs", name="seq2seq_en_es")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch{epoch}-valloss{val_loss:.3f}",
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=15,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        callbacks=[checkpoint_cb],
        logger=logger,
        log_every_n_steps=10,
        precision="16-mixed",
        num_sanity_val_steps=0,
        gradient_clip_val=1.0
    )

    # Resume-from-last if it exists
    ckpt = "checkpoints/last.ckpt"
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt if os.path.exists(ckpt) else None
    )

    print("Training finished.")
    print("Best checkpoint:", checkpoint_cb.best_model_path)
    print("Last checkpoint:", os.path.join("checkpoints", "last.ckpt"))


if __name__ == "__main__":
    main()