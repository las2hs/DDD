import csv
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def load_data(file_path):
    with open(file_path, 'r') as f:
        return f.read()

# Define a custom tokenizer that includes punctuation marks as separate tokens
def tokenizer_with_punctuation(text):
    # Remove punctuation marks from the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split the text into tokens using whitespace as the delimiter
    tokens = text.split()
    # Add back the punctuation marks as separate tokens
    tokens_with_punct = []
    for token in tokens:
        # Add the original token without punctuation marks
        tokens_with_punct.append(token)
        # Add any punctuation marks as separate tokens
        for punct in string.punctuation:
            if punct in token:
                tokens_with_punct.append(punct)
    return tokens_with_punct

def train_model(text_data, labels):
    text_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenizer_with_punctuation)), # Convert text to a bag-of-words representation
        ('classifier', MultinomialNB()) # Train a Naive Bayes classifier
    ])
    text_pipeline.fit(text_data, labels)
    return text_pipeline

def predict_new_text(model, new_text_data):
    predictions = model.predict(new_text_data)
    probabilities = model.predict_proba(new_text_data)
    return predictions, probabilities

def write_predictions_to_csv(predictions, probabilities, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Line', 'Prediction', 'CRIED Probability', 'SAID Probability'])
        for i, (prediction, probs) in enumerate(zip(predictions, probabilities)):
            writer.writerow([i+1, prediction, probs[0], probs[1]])

def verb_classifier():
    text_data_1 = load_data('./corpora/cried.txt')
    text_data_2 = load_data('./corpora/said.txt')
    text_data = [text_data_1, text_data_2]
    labels = ['CRIED', 'SAID']
    model = train_model(text_data, labels)
    new_text_data = load_data('./corpora/else.txt').split('\n')
    predictions, probabilities = predict_new_text(model, new_text_data)
    write_predictions_to_csv(predictions, probabilities, 'predictions.csv')

if __name__ == '__main__':
    verb_classifier()
