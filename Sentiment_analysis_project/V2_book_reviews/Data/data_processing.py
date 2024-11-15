import pandas as pd
from fractions import Fraction
import pickle


"""
Data_storage cleaning :
0 helpfulness
review summary with only ids
emtpy review ?
"""


def data_processing(csv_path, n_rows, min_helpfulness, default_helpfulness):
    review_helpfulness, review_score, review_summary, review_text = load_data(csv_path, n_rows)
    review_helpfulness, review_score, review_summary, review_text = clean_data(review_helpfulness, review_score,
                                                                               review_summary, review_text,
                                                                               min_helpfulness, default_helpfulness)

    # Create a dictionary
    data = {
        "helpfulness": review_helpfulness,
        "score": review_score,
        "summary": review_summary,
        "text": review_text
    }

    return data


def clean_data(review_helpfulness, review_score, review_summary, review_text, min_helpfulness, default_helpfulness):
    review_helpfulness, review_score, review_summary, review_text = remove_no_helpfulness(review_helpfulness,
                                                                                          review_score, review_summary,
                                                                                          review_text, min_helpfulness,
                                                                                          default_helpfulness)
    review_score = check_score(review_score)
    review_score = one_hot_encode(review_score)

    return review_helpfulness, review_score, review_summary, review_text


def check_score(review_score):
    # Assume `predictions` is your list of continuous values
    new_review_score = [max(1, min(5, int(round(p)))) for p in review_score]

    return new_review_score


def one_hot_encode(scores):
    # Initialize an empty list for the 2D one-hot encoded list
    one_hot_list = []

    for score in scores:
        # Create a list of zeros for each score
        one_hot = [0] * 5
        # Set the position corresponding to the score (score - 1) to 1
        one_hot[score - 1] = 1
        # Append this one-hot encoded row to the 2D list
        one_hot_list.append(one_hot)

    return one_hot_list

def remove_no_helpfulness(review_helpfulness, review_score, review_summary, review_text, min_helpfulness, default_helpfulness):
    # Convert each fraction string to a float, with a special case for "0/0"
    """
    review_helpfulness_float = [
        # If the review have 0 helpfulness reviews, we arbitrarily say that it has 1/2
        float(Fraction(helpfulness) if helpfulness != "0/0" else Fraction("1/2"))
        for helpfulness in review_helpfulness
    ]
    """
    review_helpfulness_float = [
        # If the review have 0 helpfulness reviews, we arbitrarily say that it has 1/2
        float(Fraction(helpfulness) if helpfulness != "0/0" else Fraction(default_helpfulness))
        for helpfulness in review_helpfulness
    ]

    # Create a mask for reviews with helpfulness greater than 0
    mask = [helpfulness > min_helpfulness for helpfulness in review_helpfulness_float]

    # Filter the reviews based on the mask
    filtered_review_helpfulness = [helpfulness for helpfulness, keep in zip(review_helpfulness_float, mask) if keep]
    filtered_review_score = [score for score, keep in zip(review_score, mask) if keep]
    filtered_review_summary = [summary for summary, keep in zip(review_summary, mask) if keep]
    filtered_review_text = [text for text, keep in zip(review_text, mask) if keep]

    return filtered_review_helpfulness, filtered_review_score, filtered_review_summary, filtered_review_text


def load_data(file_path, n_rows):
    print("Loading Data...")
    df = pd.read_csv(file_path, nrows=n_rows)

    # Convert each column to a separate list
    review_helpfulness = df.iloc[:, 5].tolist()  # Review helpfulness
    review_score = df.iloc[:, 6].tolist()  # Review score
    review_summary = df.iloc[:, 8].tolist()  # Review summary
    review_text = df.iloc[:, 9].tolist()  # Review text

    print("Data_storage loaded.")

    return review_helpfulness, review_score, review_summary, review_text


def save_to_pickle(data, filename='reviews.pkl'):
    with open(filename, 'wb') as f:
        # Save the lists as a tuple
        pickle.dump(data, f)

    print("Data_storage saved at : ", filename)


def load_from_pickle(filename='reviews.pkl'):
    with open(filename, 'rb') as f:
        # Load the lists from the pickle file
        data = pickle.load(f)
    return data
