# translator_module.py
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import kagglehub

from models.Seq2Seq import Seq2Seq
from models.EncoderTrans import EncoderTrans
from models.DecoderTrans import DecoderTrans


# -------------------------
# Tokenization
# -------------------------
TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:'[A-Za-z]+)?|[.,!?;:]", flags=0)

def fast_tok(s: str):
    return TOKEN_RE.findall(s)


# -------------------------
# Helpers
# -------------------------
def ids_to_text(ids, id2word, pad_id=None, eos_id=None, bos_id=None):
    words = []
    for i in ids:
        i = int(i)
        if pad_id is not None and i == pad_id:
            continue
        if bos_id is not None and i == bos_id:
            continue
        if eos_id is not None and i == eos_id:
            break
        words.append(id2word.get(i, "<UNK>"))
    return " ".join(words)


def build_dicts(english_tokens, spanish_tokens):
    # English dict
    en2id = {}
    for sen in english_tokens:
        for w in sen:
            if w not in en2id:
                en2id[w] = len(en2id)
    en2id["<EOS>"] = len(en2id)
    PAD_EN = len(en2id); en2id["<PAD>"] = PAD_EN
    en2id["<UNK>"] = len(en2id)
    UNK_EN = en2id["<UNK>"]

    # Spanish dict
    es2id = {}
    for sen in spanish_tokens:
        for w in sen:
            if w not in es2id:
                es2id[w] = len(es2id)
    es2id["<EOS>"] = len(es2id)
    es2id["<BOS>"] = len(es2id)
    PAD_ES = len(es2id); es2id["<PAD>"] = PAD_ES

    id2en = {i: w for w, i in en2id.items()}
    id2es = {i: w for w, i in es2id.items()}

    return en2id, es2id, id2en, id2es, PAD_EN, PAD_ES, UNK_EN


# -------------------------
# Main load function
# -------------------------
def load_translator(
    ckpt_path="checkpoints/last.ckpt",
    kaggle_dataset="lonnieqin/englishspanish-translation-dataset",
    csv_name="data.csv",
    embedding_dim=256,
):
    # 1) Load dataset (for vocab rebuild)
    dataset_path = kagglehub.dataset_download(kaggle_dataset)
    csv_path = os.path.join(dataset_path, csv_name)
    data = pd.read_csv(csv_path)

    english_tokens = [fast_tok(s) for s in data["english"]]
    spanish_tokens = [fast_tok(s) for s in data["spanish"]]

    en2id, es2id, id2en, id2es, PAD_EN, PAD_ES, UNK_EN = build_dicts(
        english_tokens, spanish_tokens
    )

    # 2) Build embeddings
    vocab_size_en = len(en2id)
    vocab_size_es = len(es2id)

    english_emb = nn.Embedding(vocab_size_en, embedding_dim, padding_idx=PAD_EN)
    spanish_emb = nn.Embedding(vocab_size_es, embedding_dim, padding_idx=PAD_ES)

    # 3) Build model + load checkpoint
    encoder = EncoderTrans(english_emb, PAD_EN)
    decoder = DecoderTrans(spanish_emb, PAD_ES)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = Seq2Seq.load_from_checkpoint(
        ckpt_path,
        encoder=encoder,
        decoder=decoder,
        pad_en=PAD_EN,
        pad_es=PAD_ES,
        eos_es=es2id["<EOS>"],
        bos_es=es2id["<BOS>"],
        lr=1e-3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "en2id": en2id,
        "id2es": id2es,
        "PAD_ES": PAD_ES,
        "EOS_ES": es2id["<EOS>"],
        "BOS_ES": es2id["<BOS>"],
        "UNK_EN": UNK_EN,
    }


# -------------------------
# Translate function
# -------------------------
@torch.no_grad()
def translate_text(bundle, sentence: str, use_beam=True, beam_size=3, max_len=50, len_norm_alpha=0.3):
    model = bundle["model"]
    en2id = bundle["en2id"]
    id2es = bundle["id2es"]
    PAD_ES = bundle["PAD_ES"]
    EOS_ES = bundle["EOS_ES"]
    BOS_ES = bundle["BOS_ES"]
    UNK_EN = bundle["UNK_EN"]

    device = next(model.parameters()).device

    tokens = fast_tok(sentence)
    src_ids = torch.tensor(
        [en2id.get(t, UNK_EN) for t in tokens] + [en2id["<EOS>"]],
        dtype=torch.long,
        device=device
    )

    if use_beam:
        pred_ids = model.beam_search(src_ids, beam_size, max_len, len_norm_alpha)
    else:
        pred_ids = model.greedy_decode(src_ids, max_len=max_len)

    pred_ids = pred_ids.reshape(-1)
    return ids_to_text(pred_ids, id2es, pad_id=PAD_ES, eos_id=EOS_ES, bos_id=BOS_ES)