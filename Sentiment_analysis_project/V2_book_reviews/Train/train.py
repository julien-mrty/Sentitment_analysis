import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def train_validate_model(model, dataloader, num_epochs, criterion, optimizer, training_logger):
    print("Beginning of training...\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):

        """ TRAIN MODEL """
        train_epoch_loss, train_model_output, train_target_values = train_one_epoch(model, dataloader, criterion, optimizer, device)
        train_avg_loss = train_epoch_loss / len(dataloader)
        print(f"TRAIN : Epoch {epoch + 1}, Loss: {train_avg_loss:.4f}")

        # Convert vectors to class labels
        train_target_labels = np.argmax(train_target_values, axis=1)
        train_model_labels = np.argmax(train_model_output, axis=1)
        num_labels = max(train_target_labels)

        print("train_target_labels : ", train_target_labels)
        print("train_model_labels : ", train_model_labels)

        train_report = classification_report(train_target_labels, train_model_labels, labels=range(1, num_labels + 1),
                                                                      zero_division=0, output_dict=True)
        train_confusion_matrix = confusion_matrix(train_target_labels, train_model_labels)


        """ TEST MODEL """
        val_loss, val_model_predictions, val_target_values = validate_model(model, dataloader, criterion, device)
        val_avg_loss = val_loss / len(dataloader)
        print(f"===> VALIDATION : Epoch {epoch + 1}, Loss: {val_avg_loss:.4f}")

        # Convert vectors to class labels
        val_target_labels = np.argmax(val_target_values, axis=1)
        val_model_labels = np.argmax(val_model_predictions, axis=1)
        num_labels = max(val_target_labels)

        validation_report = classification_report(val_target_labels, val_model_labels, labels=range(1, num_labels + 1),
                                                                      zero_division=0, output_dict=True)
        val_confusion_matrix = confusion_matrix(val_target_labels, val_model_labels)

        # Log the epoch information
        training_logger.log_epoch(epoch, train_avg_loss, train_report, val_avg_loss, validation_report,
                                  train_confusion_matrix, val_confusion_matrix)

    return model, training_logger


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    model_output_list = []
    target_values_list = []

    for input_ids, attention_mask, review_helpfulness, score in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        review_helpfulness = review_helpfulness.to(device)
        score = score.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, review_helpfulness)
        loss = criterion(outputs, score)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Detach and convert to numpy for sklearn compatibility
        model_output_list.extend(outputs.detach().cpu().numpy())
        target_values_list.extend(score.detach().cpu().numpy())

    return epoch_loss, model_output_list, target_values_list


def validate_model(model, dataloader, criterion, device):
    model.eval()

    val_loss = 0
    model_predictions = []
    target_values = []

    with torch.no_grad():
        for input_ids, attention_mask, review_helpfulness, score in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            review_helpfulness = review_helpfulness.to(device)
            score = score.to(device)

            outputs = model(input_ids, attention_mask, review_helpfulness)
            loss = criterion(outputs, score)

            val_loss += loss.item()
            model_predictions.extend(outputs)
            target_values.extend(score)

            # Detach and convert to numpy for sklearn compatibility
            model_predictions.extend(outputs.detach().cpu().numpy())
            target_values.extend(score.detach().cpu().numpy())

    return val_loss, model_predictions, target_values
