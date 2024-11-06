import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class ReviewModel(nn.Module):
    def __init__(self):
        super(ReviewModel, self).__init__()
        # Text feature extraction with BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_hidden_size = self.bert.config.hidden_size

        # Numerical feature layers
        self.num_features_layer = nn.Sequential(
            nn.Linear(2, 64),  # For review_helpfulness and review_score
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Assuming 3 classes: Positive, Neutral, Negative
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        # Process text features through BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output  # Shape: (batch_size, bert_hidden_size)

        # Process numerical features through dense layers
        num_features = self.num_features_layer(numerical_features)

        # Concatenate text and numerical features
        combined_features = torch.cat((text_features, num_features), dim=1)

        # Final classification
        output = self.classifier(combined_features)
        return output


# Example usage:
# Initialize model, tokenizer, and define inputs (tokenize text data, prepare numerical features)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = ReviewModel()

# Assume `texts` contains review summaries/texts, and `num_data` contains helpfulness and score
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
numerical_data = torch.tensor(num_data, dtype=torch.float32)

# Forward pass through the model
logits = model(inputs['input_ids'], inputs['attention_mask'], numerical_data)


"""INPUT"""

from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input text
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors='pt')

input_ids = inputs['input_ids']           # Input IDs
attention_mask = inputs['attention_mask'] # Attention Mask

