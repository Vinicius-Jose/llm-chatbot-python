import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq

# Create the LLM
llm = ChatOllama(model=st.secrets.get("LLM_MODEL", "llama3.1"))
if st.secrets.get("GROQ_MODEL"):
    llm = ChatGroq(model=st.secrets.get("GROQ_MODEL", "deepseek-r1-distill-llama-70b"))

# Create the Embedding model
embeddings = OllamaEmbeddings(
    model=st.secrets.get("EMBEDDINGS", "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16")
)
