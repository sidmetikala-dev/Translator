# Transformer-based English → Spanish Translator

This project implements a custom Transformer encoder–decoder from scratch using PyTorch (no `nn.Transformer` or high-level attention APIs).  
The model is trained on 118k sentence pairs and exposed via a Flask REST API for real-time inference, with MySQL-backed persistence for user tracking and feedback logging.

## Model Architecture

Implemented from scratch:
- Multi-head self-attention
- Masked decoder self-attention
- Encoder–decoder cross-attention
- Sinusoidal positional encodings

Training includes padding-aware cross-entropy loss, teacher forcing, and beam search decoding.

## Results

- Dataset: 118,964 English–Spanish sentence pairs
- Test cross-entropy: ~2.17
- Padding-masked token accuracy: ~63%

## System Architecture

Client → Flask REST API → Transformer Model → MySQL

- Model weights are loaded once at application startup from a saved checkpoint.
- `/translate` endpoint accepts JSON input and returns translated text.
- MySQL stores users, translation history, and structured feedback.
- Cookies are used for lightweight user persistence.

## Running the API

pip install -r requirements.txt  
python app.py  
POST to /translate with JSON body:
{ "text": "Hello world" }
