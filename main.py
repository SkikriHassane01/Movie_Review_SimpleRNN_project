# Import the libraries
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model

import streamlit as st
# load the imdb dataset 
max_features = 10000 # size of vocabulary (number of unique words in your dataset)
(X_train, y_train),(X_test,y_test) = imdb.load_data(num_words = max_features)


# mapping of words index to words
word_index = imdb.get_word_index()
reversed_word_index = {value:key for key,value in word_index.items()}

# load the model
model = load_model('SimpleRNN_mdb.h5')


## Helper functions
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i -3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)

    return padded_review

# prediction function 

def prediction(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]    

# streamlit app 
st.title('IMDB movie review sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# user input 
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    sentiment, score = prediction(user_input)
    
    # display the results 
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write(f'Please Enter a movie review.')
    