import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import gzip
import pandas as pd
from sklearn.model_selection import train_test_split

# Load FastText word embeddings for Hindi
# def load_embeddings(embedding_path='cc.hi.300.vec.gz'):
#     embedding_matrix = {}
#     try:
#         with gzip.open(embedding_path, 'rt', encoding='utf-8') as f:
#             for line in f:
#                 values = line.split()
#                 word = values[0]
#                 coefs = np.asarray(values[1:], dtype='float32')
#                 embedding_matrix[word] = coefs
#     except Exception as e:
#         print(f"Error: {e}")
#     return embedding_matrix

def preprocess_text(text, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

# def provide_feedback(sentence, is_sarcastic):
#     # Record the feedback in a CSV file
#     with open('feedback.csv', 'a', encoding='utf-8') as f:
#         f.write(f"{sentence},{is_sarcastic}\n")
#     st.success("Feedback has been recorded. Thank you!")

def retrain_model(sentence, is_sarcastic, tokenizer, max_length, model):
    # Convert the single sentence for retraining
    preprocessed_sentence = preprocess_text(sentence, tokenizer, max_length)
    X_train = preprocessed_sentence
    y_train = np.array([is_sarcastic])
    
    # Retrain the model with the new data
    model.fit(X_train, y_train, epochs=1, batch_size=1)
    model.save('hindi_model.h5')

def main():
    st.title("Sarcasm Detection in Hindi Tweets")

    # Load model
    model = load_model('hindi_model.h5')
    
    # Load sarcastic tweets
    sarcastic_data = pd.read_csv('Sarcasm_Hindi_Tweets-SARCASTIC.csv', encoding='utf-8')

    # Load non-sarcastic tweets
    non_sarcastic_data = pd.read_csv('Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv', encoding='utf-8')

    # Assign labels (1 for sarcastic, 0 for non-sarcastic)
    sarcastic_data['label'] = 1
    non_sarcastic_data['label'] = 0

    # Concatenate the datasets
    data = pd.concat([sarcastic_data, non_sarcastic_data], ignore_index=True)

    # Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    max_length = 100  # Make sure this matches the max_length used during training

    user_input = st.text_input("Enter a Hindi sentence:")
    if user_input:
        processed_input = preprocess_text(user_input, tokenizer, max_length)
        prediction = model.predict(processed_input)
        sarcasm_probability = prediction[0][0]

        # Display prediction
        predicted_label = 1 if sarcasm_probability > 0.5 else 0
        st.write(f"Prediction: This sentence is {'sarcastic' if predicted_label == 1 else 'not sarcastic'}")

        # Ask for user feedback
        feedback = st.radio("Was the prediction correct?", ("Yes", "No"))
        if feedback.lower() == "no":
            # If feedback is 'no', use the opposite label of the predicted one
            corrected_label = 1 - predicted_label
            # provide_feedback(user_input, corrected_label)
            retrain_model(user_input, corrected_label, tokenizer, max_length, model)
            st.success("The model has been updated with your feedback.")

if __name__ == "__main__":
    main()
