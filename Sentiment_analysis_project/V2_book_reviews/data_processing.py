import pandas as pd
from fractions import Fraction
import pickle


"""
Data cleaning :
0 helpfulness
review summary with only ids
emtpy review ?
"""


def data_processing(csv_path, n_rows):
    review_helpfulness, review_score, review_summary, review_text = load_data(csv_path, n_rows)
    review_helpfulness, review_score, review_summary, review_text = clean_data(review_helpfulness, review_score, review_summary, review_text)

    return review_helpfulness, review_score, review_summary, review_text


def clean_data(review_helpfulness, review_score, review_summary, review_text):
    review_helpfulness, review_score, review_summary, review_text = remove_no_helpfulness(review_helpfulness, review_score, review_summary, review_text)
    review_score = normalize_scores(review_score)

    for i in range(len(review_score)):
            print("===============review_score :", review_score[i])
            print("review_helpfulness :", review_helpfulness[i])
            print("review_summary :", review_summary[i])
            print("review_text :", review_text[i])

    return review_helpfulness, review_score, review_summary, review_text


def normalize_scores(review_score):
    min_score = min(review_score)
    max_score = max(review_score)

    # Avoid division by zero
    if max_score - min_score == 0:
        normalized_scores = [0.5] * len(review_score)  # Or handle it differently as needed
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in review_score]

    return normalized_scores


def remove_no_helpfulness(review_helpfulness, review_score, review_summary, review_text):
    # Convert each fraction string to a float, with a special case for "0/0"
    review_helpfulness_float = [
        # If the review have 0 helpfulness reviews, we arbitrarily say that it has 1/2
        float(Fraction(helpfulness) if helpfulness != "0/0" else Fraction("1/2"))
        for helpfulness in review_helpfulness
    ]

    # Create a mask for reviews with helpfulness greater than 0
    mask = [helpfulness > 0 for helpfulness in review_helpfulness_float]

    # Filter the reviews based on the mask
    filtered_review_helpfulness = [helpfulness for helpfulness, keep in zip(review_helpfulness_float, mask) if keep]
    filtered_review_score = [score for score, keep in zip(review_score, mask) if keep]
    filtered_review_summary = [summary for summary, keep in zip(review_summary, mask) if keep]
    filtered_review_text = [text for text, keep in zip(review_text, mask) if keep]

    return filtered_review_helpfulness, filtered_review_score, filtered_review_summary, filtered_review_text


def load_data(file_path, n_rows):
    print("Loading data...")
    df = pd.read_csv(file_path, nrows=n_rows)

    print(df.iloc[0].tolist())

    # Convert each column to a separate list
    review_helpfulness = df.iloc[:, 5].tolist()  # Review helpfulness
    review_score = df.iloc[:, 6].tolist()  # Review score
    review_summary = df.iloc[:, 8].tolist()  # Review summary
    review_text = df.iloc[:, 9].tolist()  # Review text

    print("Data loaded.")

    return review_helpfulness, review_score, review_summary, review_text


def save_to_pickle(review_helpfulness, review_score, review_summary, review_text, filename='reviews.pkl'):
    with open(filename, 'wb') as f:
        # Save the lists as a tuple
        pickle.dump((review_helpfulness, review_score, review_summary, review_text), f)


def load_from_pickle(filename='reviews.pkl'):
    with open(filename, 'rb') as f:
        # Load the lists from the pickle file
        review_helpfulness, review_score, review_summary, review_text = pickle.load(f)
    return review_helpfulness, review_score, review_summary, review_text
