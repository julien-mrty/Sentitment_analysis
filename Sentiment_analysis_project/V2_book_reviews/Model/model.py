import torch.nn as nn
import torch
from transformers import BertModel
from transformers import DistilBertModel


class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze the first 4 layers
        for layer in self.distilbert.transformer.layer[:4]:
            for param in layer.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=0.3)
        self.fc_helpfulness = nn.Linear(1, 32)  # For review_helpfulness
        self.fc_combined = nn.Linear(768 + 32, 5)  # Output 5 classes

    def forward(self, input_ids, attention_mask, review_helpfulness):
        # DistilBERT encoding for text
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        # Use the [CLS] token representation (first token)
        cls_token = hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Apply dropout
        pooled_output = self.dropout(cls_token)

        # Process review_helpfulness feature
        helpfulness_output = torch.relu(self.fc_helpfulness(review_helpfulness.unsqueeze(1)))

        # Concatenate BERT output with helpfulness feature
        combined_output = torch.cat((pooled_output, helpfulness_output), dim=1)

        # Final classification layer
        logits = self.fc_combined(combined_output)

        # No Softmax in Output: Return raw logits suitable for nn.CrossEntropyLoss().
        return logits


class SentimentAnalysisModel_V0(nn.Module):
    def __init__(self):
        super(SentimentAnalysisModel_V0, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc_helpfulness = nn.Linear(1, 32)  # for review_helpfulness
        self.fc_combined = nn.Linear(768 + 32, 5)  # output 5 classes instead of 1

    def forward(self, input_ids, attention_mask, review_helpfulness):
        # BERT encoding for text
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # (batch_size, 768)

        # Dense layer for helpfulness
        helpfulness_output = torch.relu(self.fc_helpfulness(review_helpfulness.unsqueeze(1)))

        # Concatenate and final layer
        combined_output = torch.cat((pooled_output, helpfulness_output), dim=1)
        output = self.fc_combined(combined_output)

        # No Softmax in Output: Return raw logits suitable for nn.CrossEntropyLoss().
        return output


    """
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc_helpfulness = nn.Linear(1, 32)  # for review_helpfulness
        self.fc_combined = nn.Linear(768 + 32, 5)  # output 5 classes instead of 1

    def forward(self, input_ids, attention_mask, review_helpfulness):
        # BERT encoding for text
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # (batch_size, 768)

        # Dense layer for helpfulness
        helpfulness_output = torch.relu(self.fc_helpfulness(review_helpfulness.unsqueeze(1)))

        # Concatenate and final layer
        combined_output = torch.cat((pooled_output, helpfulness_output), dim=1)
        output = self.fc_combined(combined_output)

        # No Softmax in Output: Return raw logits suitable for nn.CrossEntropyLoss().
        return output
    """