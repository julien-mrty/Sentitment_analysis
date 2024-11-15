from V2_book_reviews.dataset import AmazonReviewDataset
from V2_book_reviews import paths
from V2_book_reviews.Data import data_processing
from V2_book_reviews import logger
from V2_book_reviews.Train import train, train_tools
import os
from torch.utils.data import DataLoader
from V2_book_reviews.model import SentimentAnalysisModel
import torch.nn as nn
import torch.optim as optim


"""
Dataset : https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
"""
training_csv_path = paths.training_csv_path


""" Data_storage parameters """
n_rows = 100
min_helpfulness = -1 # Minimum value of helpfulness, between 0 and 1
default_helpfulness = 1e-3 # If the review has zero helpfulness review
data_filename = paths.data_filename


""" Hyperparameters """
batch_size = 16
shuffle = True
learning_rate = 1e-5
num_epochs = 1


def main():
    if not (os.path.isfile(data_filename)):
        print("Data_storage extraction from csv...")
        data = data_processing.data_processing(training_csv_path, n_rows, min_helpfulness, default_helpfulness)
        data_processing.save_to_pickle(data, filename=data_filename)
    else:
        print("Data_storage already extracted.\n")

    data = data_processing.load_from_pickle(filename=data_filename)

    # Create dataset and dataloader
    train_dataset = AmazonReviewDataset(data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize model, loss function, and optimizer
    print("Model initialisation...\n")
    model = SentimentAnalysisModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_logger = logger.ModelTrainingLogger()

    model, training_logger = train.train_validate_model(model, train_data_loader, num_epochs, criterion, optimizer, training_logger)

    train_tools.plot_training_results(training_logger)

if __name__ == '__main__':
    main()