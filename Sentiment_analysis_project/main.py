import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import re
import numpy as np

# Sample dataset (you should replace this with your actual dataset)
texts = ["I love this product", "This is terrible", "Amazing experience", "I hate it", "Best purchase ever",
         "Not worth it"]
labels = [1, 0, 1, 0, 1, 0]


# Text preprocessing: tokenization and padding
def preprocess_texts(texts, vocab_size=5000, max_length=10):
    all_words = [word for text in texts for word in re.findall(r'\b\w+\b', text.lower())]
    word_counts = Counter(all_words)
    vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(vocab_size - 1))}

    def encode(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [vocab.get(word, 0) for word in words][:max_length] + [0] * (max_length - len(words))

    return [encode(text) for text in texts], vocab


# Prepare data
X, vocab = preprocess_texts(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)


# Model definition
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 10, hidden_dim),  # 10 is max length
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)  # Flatten
        return self.fc(x)


# Parameters
vocab_size = len(vocab) + 1  # Account for padding index
embed_dim = 16
hidden_dim = 32
model = SentimentModel(vocab_size, embed_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts).squeeze()
        preds = (outputs >= 0.5).float()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy * 100:.2f}%')
