from V1_twitter import data_processing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

training_csv_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/AI/Datasets/twitter_training.csv"
validation_csv_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/AI/Datasets/twitter_validation.csv"


def train(sentiments_list, messages_list):
    # Convert text Data to numerical features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(messages_list)  # Transform phrases into a bag-of-words representation

    # Split Data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, sentiments_list, test_size=0.1, random_state=42)

    # Using set to get unique values
    unique_prediction_values = set(y_train)
    # Print unique values
    print("Possible prediction values : ", unique_prediction_values)

    # Initialize and Train the Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return model, vectorizer  # Return model and vectorizer for later use


def predict_phrase(model, vectorizer):
    while True:
        my_phrase = input("Enter a phrase (or 'exit' to quit): ")
        if my_phrase.lower() == 'exit':
            break

        my_phrase_vec = vectorizer.transform([my_phrase])  # Transform the input phrase
        prediction = model.predict(my_phrase_vec)
        print("Sentiment:", prediction[0])


def main():
    train_sentiments_list, train_messages_list_unicode = data_processing.data_pre_processing(training_csv_path)
    val_sentiments_list, val_messages_list_unicode = data_processing.data_pre_processing(validation_csv_path)

    # Merge the Train and split. We will automatically split and shuffle the Data afterward in the Train method.
    sentiments_list = train_sentiments_list + val_sentiments_list
    messages_list_unicode = train_messages_list_unicode + val_messages_list_unicode

    model, vectorizer = train(sentiments_list, messages_list_unicode)  # Get the trained model and vectorizer

    predict_phrase(model, vectorizer)  # Start the prediction loop


if __name__ == '__main__':
    main()