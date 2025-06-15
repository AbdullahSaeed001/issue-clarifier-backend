"""
Model loading and inference functions for issue classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.config import BUG_MODEL_NAME, FEATURE_MODEL_NAME, BUG_LABELS, FEATURE_LABELS

def load_models(hyperparams):
    """Load and initialize the bug and feature classification models."""
    # Bug classification model
    bug_tokenizer = AutoTokenizer.from_pretrained(BUG_MODEL_NAME)
    bug_model = AutoModelForSequenceClassification.from_pretrained(BUG_MODEL_NAME, num_labels=2)
    bug_model.eval()
    
    # Feature classification model
    feature_tokenizer = AutoTokenizer.from_pretrained(FEATURE_MODEL_NAME)
    feature_model = AutoModelForSequenceClassification.from_pretrained(FEATURE_MODEL_NAME, num_labels=2)
    feature_model.eval()
    
    return bug_tokenizer, bug_model, feature_tokenizer, feature_model

def classify_bug(text, bug_tokenizer, bug_model, hyperparams):
    """Classify text as bug or non-bug."""
    encoded = bug_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=hyperparams["max_length"],
        return_tensors="pt"
    )
    with torch.no_grad():
        output = bug_model(**encoded)
    pred = torch.argmax(output.logits, dim=1).item()
    return BUG_LABELS[pred]

def classify_feature(text, feature_tokenizer, feature_model, hyperparams):
    """Classify text as feature or improvement."""
    encoded = feature_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=hyperparams["max_length"],
        return_tensors="pt"
    )
    with torch.no_grad():
        output = feature_model(**encoded)
    pred = torch.argmax(output.logits, dim=1).item()
    return FEATURE_LABELS[pred] 