# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# #Load the LSTM Model
# model=load_model('next_word_lstm.h5')

# #3 Laod the tokenizer
# with open('tokenizer.pickle','rb') as handle:
#     tokenizer=pickle.load(handle)

# # Function to predict the next word
# def predict_next_word(model, tokenizer, text, max_sequence_len):
#     token_list = tokenizer.texts_to_sequences([text])[0]
#     if len(token_list) >= max_sequence_len:
#         token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
#     token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
#     predicted = model.predict(token_list, verbose=0)
#     predicted_word_index = np.argmax(predicted, axis=1)
#     for word, index in tokenizer.word_index.items():
#         if index == predicted_word_index:
#             return word
#     return None

# # streamlit app
# st.title("Next Word Prediction With LSTM And Early Stopping")
# input_text=st.text_input("Enter the sequence of Words","To be or not to")
# if st.button("Predict Next Word"):
#     max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
#     next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
#     st.write(f'Next word: {next_word}')


# app.py
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM as KerasLSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom LSTM that drops unsupported args
class LSTMIgnore(KerasLSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# When loading, map 'LSTM' to your shim:
model = load_model(
    'next_word_lstm.h5',
    custom_objects={'LSTM': LSTMIgnore},
    compile=False
)

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

# …the rest of your predict & Streamlit code…

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
