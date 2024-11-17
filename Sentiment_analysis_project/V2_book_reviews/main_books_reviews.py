import torch
from V2_book_reviews.dataset import AmazonReviewDataset
from V2_book_reviews import paths
from V2_book_reviews.Data import data_processing
from V2_book_reviews import logger
from V2_book_reviews.Train import train, train_tools
import os
from torch.utils.data import DataLoader
from V2_book_reviews.Model.model import SentimentAnalysisModel
from V2_book_reviews.Model import model_tools
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

""" Dataset : https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews """
# Get the paths depending on the environment
training_csv_path, data_filename, results_rectory = paths.get_paths()

""" Data_storage parameters """
n_samples = 600000
min_helpfulness = -1  # Minimum value of helpfulness, between 0 and 1
default_helpfulness = 1e-3  # If the review has no helpfulness review
data_filename += str(n_samples)  # Update the name with the current number of samples
five_stars_reviews_percentage_to_remove = 0.85
four_stars_reviews_percentage_to_remove = 0.54

""" Hyperparameters """
batch_size = 32
shuffle = True
learning_rate = 2e-5
weight_decay = 3e-7
num_epochs = 1
split_ratio = 0.85
print_freq = 20  # Every X batch


def main():
    if torch.cuda.is_available():
        print("torch.cuda.device_count() : ", torch.cuda.device_count())
        print("Training on : ", torch.cuda.get_device_name(0))

    if not (os.path.isfile(data_filename)):
        print("Data_storage extraction from csv...")
        data = data_processing.data_processing(training_csv_path, n_samples, min_helpfulness, default_helpfulness)
        # Balancing the dataset
        data_processing.delete_score_reviews(data, five_stars_reviews_percentage_to_remove, 4)
        data_processing.delete_score_reviews(data, four_stars_reviews_percentage_to_remove, 3)

        data_processing.save_to_pickle(data, filename=data_filename)
    else:
        print("Data_storage already extracted.\n")

    data = data_processing.load_from_pickle(filename=data_filename)

    train_data, validation_data = data_processing.split_data(data, split_ratio)

    # Create dataset and dataloader for training and validation
    train_dataset = AmazonReviewDataset(train_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dataset = AmazonReviewDataset(validation_data)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize model
    print("Model initialisation...\n")
    model = SentimentAnalysisModel()
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()
    # Initialize the optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Define optimizer with parameter groups
    optimizer = optim.Adam([
        {'params': model.distilbert.parameters(), 'lr': 5e-6},  # Lower learning rate for DistilBERT
        {'params': model.fc_helpfulness.parameters(), 'lr': 1e-4},  # Higher learning rate for new layers
        {'params': model.fc_combined.parameters(), 'lr': 1e-4}
    ], weight_decay=weight_decay)
    # Initialize learning rate scheduler
    total_steps = len(train_data_loader) * num_epochs
    num_warmup_steps = int(0.15 * total_steps)  # 10% of total steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # Create training logger for the model's training
    training_logger = logger.ModelTrainingLogger()

    # Train the model
    model, training_logger = train.train_validate_model(model, train_data_loader, validation_data_loader, num_epochs,
                                                        criterion, optimizer, scheduler, training_logger, print_freq)
    # Save the model and logger
    model_filename, logger_filename = save_model_logger(model, training_logger)

    # Load the model
    model_loaded = SentimentAnalysisModel()
    model_loaded.load_state_dict(torch.load(model_filename, weights_only=True))

    # Load the logger and print training results
    training_logger_loaded = logger.ModelTrainingLogger()
    training_logger_loaded.load_from_file(logger_filename)
    train_tools.plot_training_results(training_logger_loaded)

    model_tools.use_model(model_loaded)


def save_model_logger(model, training_logger):
    # Get info training to set the name of the model and the logger
    name = data_processing.get_name(n_samples, split_ratio, num_epochs, learning_rate)
    # name += model.bert

    # Save model the state dictionary
    model_filename = results_rectory + "Model_" + name + "_state_dict.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"\nModel saved successfully at : {model_filename}\n")

    # Save the model's training logs
    logger_filename = results_rectory + "Logger_" + name + ".pkl"
    training_logger.save_to_file(logger_filename)

    return model_filename, logger_filename


if __name__ == '__main__':
    main()
