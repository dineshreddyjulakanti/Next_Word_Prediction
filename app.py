# app.py

# ─── SHIM for your old pickle’s module path ─────────────────────────────────────
import sys, types
import tensorflow.keras.preprocessing.text as ktext

# Create placeholder modules so pickle can find keras.src.preprocessing.text.Tokenizer
sys.modules['keras'] = types.ModuleType('keras')
sys.modules['keras.src'] = types.ModuleType('keras.src')
sys.modules['keras.src.preprocessing'] = types.ModuleType('keras.src.preprocessing')
txt_mod = types.ModuleType('keras.src.preprocessing.text')
txt_mod.Tokenizer = ktext.Tokenizer
sys.modules['keras.src.preprocessing.text'] = txt_mod
# ────────────────────────────────────────────────────────────────────────────────

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

# Load your model (shim handles the LSTM issue)
model = load_model(
    'next_word_lstm.h5',
    custom_objects={'LSTM': LSTMIgnore},
    compile=False
)

# Unpickle your tokenizer (now succeeds because of the shim above)
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

# …the rest of your predict & Streamlit code…

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],
                               maxlen=max_sequence_len-1,
                               padding='pre')
    preds = model.predict(token_list, verbose=0)
    next_index = np.argmax(preds, axis=1)[0]
    for word, idx in tokenizer.word_index.items():
        if idx == next_index:
            return word
    return None

st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
