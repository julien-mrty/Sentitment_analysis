import torch.nn as nn
import torch
from transformers import BertModel


class SentimentAnalysisModel(nn.Module):
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

        # Apply softmax for probability distribution across 5 classes
        return output