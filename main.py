import streamlit as st
import pandas as pd
from transformers import pipeline, set_seed

st.title("Text Generation Model")

# Loading Model
with st.spinner('Working on it ...'):
    generator = pipeline('text-generation', model='gpt2')
st.success('Hurrah, Ready to go!!')


# Processing Input
with st.form("Model Input"):
    st.text_input("Input to Model", key="input")
    slider_val = st.select_slider("Specify Response Length", options=[
                                  10, 15, 20, 25, 30])
    num_responses = st.selectbox("Number of responses:", (1, 2, 3, 4, 5))
    submitted = st.form_submit_button("Generate Text")
    if submitted:
        generated_text = generator(st.session_state.input, max_length=int(
            slider_val), num_return_sequences=num_responses)
        container = st.container()
        responses = list()
        for response in generated_text:
            responses.append(response["generated_text"])
        df = pd.DataFrame({
            'Responses': responses
        })
        df.index = df.index + 1
        st.dataframe(df)
