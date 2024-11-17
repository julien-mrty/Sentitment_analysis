from V2_book_reviews import paths
from V2_book_reviews.Data import data_processing
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


""" Data_storage parameters """
training_csv_path, _, _ = paths.get_paths()
n_samples = 1000000000
min_helpfulness = -1 # Minimum value of helpfulness, between 0 and 1
default_helpfulness = 1e-3 # If the review has no helpfulness review
data_filename = "../Data_storage/Data_amazon_book_reviews.pkl"
five_stars_reviews_percentage_to_remove = 0.85
four_stars_reviews_percentage_to_remove = 0.54


def main():
    if not (os.path.isfile(data_filename)):
        print("Data_storage extraction from csv...")
        data = data_processing.data_processing(training_csv_path, n_samples, min_helpfulness, default_helpfulness)

        data_processing.delete_score_reviews(data, five_stars_reviews_percentage_to_remove, 4)
        data_processing.delete_score_reviews(data, four_stars_reviews_percentage_to_remove, 3)

        data_processing.save_to_pickle(data, filename=data_filename)
    else:
        print("Data_storage already extracted.\n")

    data = data_processing.load_from_pickle(filename=data_filename)

    plot_number_of_reviews(data)


def plot_number_of_reviews(data):
    review_scores = data['score']
    review_helpfulness = data['helpfulness']

    # Count occurrences of `1` along each column
    scores_count = [sum(row[i] for row in review_scores) for i in range(len(review_scores[0]))]
    # Plot the counts
    scores = range(len(scores_count))

    # Count unique values using numpy
    unique_helpfulness, helpfulness_counts = np.unique(review_helpfulness, return_counts=True)

    # Convert to dictionaries for sorting (optional, depends on your plot needs)
    sorted_helpfulness = dict(zip(unique_helpfulness, helpfulness_counts))

    # Set up subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for Review Score
    ax1.bar(scores, scores_count, color='skyblue', width=0.1, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Review Score')
    ax1.set_ylabel('Number of Reviews')
    ax1.set_title('Number of Reviews for Each Review Score')
    ax1.set_xticks(scores)

    # Plot for Review Helpfulness (Line Plot)
    ax2.plot(list(sorted_helpfulness.keys()), list(sorted_helpfulness.values()), color='salmon', marker='o')
    ax2.set_xlabel('Helpfulness Score')
    ax2.set_ylabel('Number of Reviews')
    ax2.set_title('Number of Reviews for Each Helpfulness Score')
    ax2.set_xlim(0, 1)  # Set x-axis between 0 and 1

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('Last_plot.png')
    #plt.show()


if __name__ == '__main__':
    main()