from V2_book_reviews import paths
from V2_book_reviews.Data import data_processing
import os
import matplotlib.pyplot as plt
from collections import Counter



""" Data_storage parameters """
training_csv_path = paths.training_csv_path
n_rows = 1000000000
min_helpfulness = 0.2 # UNUSED !!!
data_filename = "../Data_storage/Data_amazon_book_reviews.pkl"


def main():
    if not (os.path.isfile(data_filename)):
        print("Data_storage extraction from csv...")
        data = data_processing.data_processing(training_csv_path, n_rows)
        data_processing.save_to_pickle(data, filename=data_filename)
    else:
        print("Data_storage already extracted.\n")

    data = data_processing.load_from_pickle(filename=data_filename)

    plot_number_of_reviews(data)


def plot_number_of_reviews(data):
    review_scores = data['score']  # This is a list
    review_helpfulness = data['helpfulness']  # This is also a list

    # Count each score and helpfulness value
    score_counts = Counter(review_scores)
    helpfulness_counts = Counter(review_helpfulness)

    # Sort the counts by score for consistency
    sorted_scores = dict(sorted(score_counts.items()))
    sorted_helpfulness = dict(sorted(helpfulness_counts.items()))

    # Set up subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for Review Score
    ax1.bar(sorted_scores.keys(), sorted_scores.values(), color='skyblue', width=0.1, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Review Score')
    ax1.set_ylabel('Number of Reviews')
    ax1.set_title('Number of Reviews for Each Review Score')
    ax1.set_xticks(list(sorted_scores.keys()))

    # Plot for Review Helpfulness (Line Plot)
    ax2.plot(list(sorted_helpfulness.keys()), list(sorted_helpfulness.values()), color='salmon', marker='o')
    ax2.set_xlabel('Helpfulness Score')
    ax2.set_ylabel('Number of Reviews')
    ax2.set_title('Number of Reviews for Each Helpfulness Score')
    ax2.set_xlim(0, 1)  # Set x-axis between 0 and 1

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


if __name__ == '__main__':
    main()