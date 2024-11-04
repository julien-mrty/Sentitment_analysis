import data_processing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

training_csv_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/AI/Datasets/twitter_training.csv"
validation_csv_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/AI/Datasets/twitter_validation.csv"


def train(sentiments_list, messages_list):
    # Convert text data to numerical features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(messages_list)  # Transform phrases into a bag-of-words representation

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, sentiments_list, test_size=0.2, random_state=42)

    # Initialize and train the Naive Bayes classifier
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
    sentiments_list, messages_list = data_processing.load_data(training_csv_path)
    messages_list_unicode = data_processing.pre_process_phrases(messages_list)

    model, vectorizer = train(sentiments_list, messages_list_unicode)  # Get the trained model and vectorizer

    predict_phrase(model, vectorizer)  # Start the prediction loop


if __name__ == '__main__':
    main()