import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer


class AmazonReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Initialize tokenizer
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.data['score'])

    def __getitem__(self, idx):
        # Tokenize text
        inputs = self.tokenizer(str(self.data['text'][idx]) + ' ' + str(self.data['summary'][idx]),
                           return_tensors="pt", padding="max_length",
                           truncation=True, max_length=256)

        # Convert helpfulness and score to tensors
        review_helpfulness = torch.tensor(self.data['helpfulness'][idx], dtype=torch.float)
        score = torch.tensor(self.data['score'][idx], dtype=torch.float)

        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), review_helpfulness, score



