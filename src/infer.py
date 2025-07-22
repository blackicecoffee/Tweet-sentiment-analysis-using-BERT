import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Setup model
bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 3).to(device)

bert_model.load_state_dict(torch.load("model/sentiment_bert.pth", weights_only=True))