import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2Model

st.title("Text Generation Model")

# Loading Model
with st.spinner('Working on it ...'):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
st.success('Hurrah, Ready to go!!')


# Processing Input
with st.form("Model Input"):
    st.text_input("Input to Model", key="input")
    submitted = st.form_submit_button("Generate Text")
    if submitted:
        encoded_input = tokenizer(st.session_state.input, return_tensors='pt')
        output = model(**encoded_input)
        st.write(output)
