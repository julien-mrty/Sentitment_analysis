import torch


def get_paths():
    if torch.cuda.is_available():
        training_csv_path = "/Datasets/Amazon_books_review/archive/Books_rating.csv"
        data_filename = f"V2_book_reviews/Data_storage/data_reviews_"
        results_rectory = "V2_book_reviews/Training_results/"
    else:
        training_csv_path = "/home/julien/Programming/AI/Datasets/Amazon_books_review/archive/Books_rating.csv"
        data_filename = f"Data_storage/data_reviews_"
        results_rectory = "Training_results/"

    return training_csv_path, data_filename, results_rectory