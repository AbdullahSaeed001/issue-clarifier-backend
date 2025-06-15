"""
Dataset handling for model training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class IssueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """Initialize the dataset with texts and labels."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def create_dataloader(texts, labels, tokenizer, hyperparams):
    """Create a DataLoader for training or evaluation."""
    dataset = IssueDataset(texts, labels, tokenizer, hyperparams["max_length"])
    return DataLoader(
        dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True
    ) 