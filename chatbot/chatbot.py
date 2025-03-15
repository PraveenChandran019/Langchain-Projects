from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

st.title('Langchain Demo')
input_text = st.text_input("Search the topic you want")

prompt =ChatPromptTemplate.from_messages(
    [
        ("system",'you are a helpful assistant. Please response to the user queries'),
        ("user","Question : {question}")
    ]
)

llm =  ChatOllama(
    model = 'llama2:latest',
    base_url = 'http://localhost:11434',
    temperature = 0.3 
)
parser = StrOutputParser()
chain = prompt | llm | parser

if input_text:
    st.write(chain.invoke({'question': input_text}))