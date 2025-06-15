"""
Model training and evaluation functions.
"""

import torch
import torch.optim as optim
from .dataset import create_dataloader

def train_model(model, tokenizer, df, label_mapping, hyperparams):
    """Train the model on the provided dataset."""
    if not {"title", "body", "label"}.issubset(df.columns):
        raise ValueError("CSV must have 'title', 'body', and 'label' columns")
    
    # Prepare data
    df["title"] = df["title"].astype(str).fillna("")
    df["body"] = df["body"].astype(str).fillna("")
    df["summary"] = df["title"] + " " + df["body"]
    df["label_int"] = df["label"].map(label_mapping)
    
    texts = df["summary"].tolist()
    labels = df["label_int"].tolist()
    
    # Create dataloader
    dataloader = create_dataloader(texts, labels, tokenizer, hyperparams)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    
    # Training loop
    for epoch in range(hyperparams["epochs"]):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{hyperparams['epochs']} - Loss: {avg_loss:.4f}")
    
    model.eval()
    return model

def evaluate_model(model, tokenizer, df, label_mapping, hyperparams):
    """Evaluate the model on the provided dataset."""
    if not {"title", "body", "label"}.issubset(df.columns):
        raise ValueError("CSV must have 'title', 'body', and 'label' columns")
    
    # Prepare data
    df["title"] = df["title"].astype(str).fillna("")
    df["body"] = df["body"].astype(str).fillna("")
    df["summary"] = df["title"] + " " + df["body"]
    df["label_int"] = df["label"].map(label_mapping)
    
    texts = df["summary"].tolist()
    labels = df["label_int"].tolist()
    
    # Create dataloader
    dataloader = create_dataloader(texts, labels, tokenizer, hyperparams)
    
    # Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total, correct = 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            labels_batch = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy} 