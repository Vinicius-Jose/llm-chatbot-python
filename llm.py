import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Create the LLM
llm = ChatOllama(model="llama3.1")

# Create the Embedding model
embeddings = OllamaEmbeddings(model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16")
