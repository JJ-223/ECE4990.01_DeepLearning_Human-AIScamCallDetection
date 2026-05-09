
"""
daptl_inference.py
Loads the fine-tuned DAPTL scam call detector and returns
a scam probability score for a given text input.
Used by the MLP fusion notebook.
"""
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DAPTL_FINAL_CKPT_DIR = "/content/drive/MyDrive/daptl_project/daptl_final_model"
CALL_MAX_LENGTH = 512

def load_scam_model(device):
    """Load tokenizer and model from Drive checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(DAPTL_FINAL_CKPT_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(DAPTL_FINAL_CKPT_DIR)
    model     = model.to(device)
    model.eval()
    return tokenizer, model

def preprocess_text(text):
    """Match training preprocessing — lowercase + punctuation strip."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_scam_probability(text, tokenizer, model, device):
    """
    Returns scam probability (float in [0,1]) for a given text.
    1.0 = definite scam, 0.0 = definite legitimate.
    Used as p_scam input to the MLP fusion layer.
    """
    text = preprocess_text(text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=CALL_MAX_LENGTH
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        prob   = torch.softmax(logits, dim=1)[0, 1].item()
    return prob
