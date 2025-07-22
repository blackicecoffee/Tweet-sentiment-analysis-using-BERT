from tqdm import tqdm
import re

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
train_df = pd.read_csv("data/Corona_NLP_train.csv")
test_df = pd.read_csv("data/Corona_NLP_test.csv")

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\n', ' ', text) # Remove newline character
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces

    return text

train_df["cleaned_text"] = train_df["OriginalTweet"].apply(clean_text)
test_df["cleaned_text"] = test_df["OriginalTweet"].apply(clean_text)

# Label encoding
sentiments = {
    "Extremely Positive": "Positive",
    "Positive": "Positive",
    "Extremely Negative": "Negative",
    "Negative": "Negative",
    "Neutral": "Neutral"
}

train_df["Sentiment"] = train_df["Sentiment"].map(sentiments)
test_df["Sentiment"] = test_df["Sentiment"].map(sentiments)

sentiment_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

train_df["label"] = train_df["Sentiment"].map(sentiment_mapping)
test_df["label"] = test_df["Sentiment"].map(sentiment_mapping)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Custom dataset
class CoronaTweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)
    
# Model initialize
bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 3).to(device)

total_params = sum(p.numel() for p in bert_model.parameters())
trainable_params = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)

print("Model: google-bert/bert-base-uncased")
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}\n")

# Create dataset
train_ds = CoronaTweetDataset(
    texts=train_df["cleaned_text"].tolist(), 
    labels=train_df["label"].tolist(), 
    tokenizer=bert_tokenizer
)

test_ds = CoronaTweetDataset(
    texts=test_df["cleaned_text"].tolist(), 
    labels=test_df["label"].tolist(), 
    tokenizer=bert_tokenizer
)

print(f"Training size: {len(train_ds)}\nTest size: {len(test_ds)}")

# Create dataloader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(train_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

print(f"Train batch: {len(train_loader)}\nValidation batch: {len(val_loader)}\nTest batch: {len(test_loader)}\n")

# Training model
optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
epochs = 3

print(f"Number of epochs: {epochs}")
print("Training...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training mode
    bert_model.train()

    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.4f}")
    
    # Evaluation mode
    bert_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
    
            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nValidation accuracy: {acc:.4f}")

# Evaluate on test set
bert_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\Test accuracy: {acc:.4f}")

# Save model
torch.save(bert_model.state_dict(), "model/sentiment_bert.pth")
