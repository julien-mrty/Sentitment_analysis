import torch
from transformers import BertTokenizer


def use_model(model):
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Inference loop
    print("Type a sentence to analyze its sentiment (type 'exit' to quit):")
    while True:
        user_input = input(">> ")  # Get user input
        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # Preprocess the input
        input_ids, attention_mask = preprocess(user_input, tokenizer)

        # Dummy value for review_helpfulness (replace with actual value if available)
        review_helpfulness = torch.tensor([0.5])  # Example: normalized helpfulness score

        # Make prediction
        with torch.no_grad():
            predictions = model(input_ids, attention_mask, review_helpfulness)

        # Get predicted class (index of max probability)
        predicted_class = torch.argmax(predictions, dim=1).item()
        print("Model prediction : ", predictions)

        print(f"Predicted Sentiment Class: {predicted_class}")


# Function to preprocess input
def preprocess(sentence, tokenizer):
    # Tokenize the sentence
    encoding = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask
