from V2_book_reviews import data_processing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

"""
Dataset : https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
"""
training_csv_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/AI/Datasets/Amazon books review dataset/Books_rating.csv"

""" Data parameters """
n_rows = 100


def train(sentiments_list, messages_list):
    print("TODO")


def main():
    review_helpfulness, review_score, review_summary, review_text = data_processing.data_processing(training_csv_path, n_rows)




if __name__ == '__main__':
    main()