from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langserve import add_routes
import uvicorn
import os


llm = ChatOllama(
    model = 'llama2:latest',
    temperature = 0.3,
    base_url = 'http://localhost:11434'
)
prompt = ChatPromptTemplate.from_template('write a poem about {topic} for a 5 year old')


app = FastAPI(
    title = 'Langchain Server',
    version = '1.0',
    description = 'A simple API Server'
)

add_routes(
    app,
    prompt | llm,
    path = '/poem'
)
if __name__ == '__main__':
    uvicorn.run(app,host = 'localhost',port = 8000)