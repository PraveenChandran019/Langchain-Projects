import requests
import streamlit as st 
def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke", json = {'input':{"topic":input_text}})
    return response.json()['output']['content']

st.title('Langchain Demo with LLAMA2 API')
input_text = st.text_input("write an poem on")

if input_text:
    st.write(get_ollama_response(input_text))