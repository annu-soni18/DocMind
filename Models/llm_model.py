# """
# Models/llm_model.py
# -------------------
# Loads and returns the LLM and Embeddings models.

# WHY @st.cache_resource?
#     Streamlit reruns the entire script on every user interaction.
#     Without caching, a new model object (and API connection) would be
#     created every time the user types a message — wasteful and slow.
#     @st.cache_resource tells Streamlit: "create this once, reuse it forever."

# INTERVIEW POINT — Why Groq for the LLM?
#     Groq runs LLaMA3 on custom LPU (Language Processing Unit) hardware.
#     It's significantly faster than OpenAI (higher tokens/second) and
#     has a generous free tier — perfect for demos and portfolio projects.

# INTERVIEW POINT — Why OpenAI for embeddings, not Groq?
#     Groq does not offer an embeddings API — only LLM inference.
#     OpenAI's text-embedding-3-small is cheap ($0.02 per million tokens),
#     produces high-quality 1536-dimensional vectors, and is widely
#     compatible with FAISS and other vector stores.
# """

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings

# # load_dotenv() reads the .env file when running locally.
# # On Render, environment variables are set directly in the dashboard
# # so load_dotenv() has no effect there — that's fine.
# load_dotenv()


# @st.cache_resource
# def load_llm() -> ChatGroq:
#     """
#     Returns a cached ChatGroq instance using LLaMA-3.3-70B.

#     temperature=0 → fully deterministic output.
#     For RAG (factual Q&A), we want the LLM to stick to the
#     provided context, not creatively riff on it.
#     """
#     return ChatGroq(
#         groq_api_key=os.getenv("GROQ_API_KEY"),
#         model_name="llama-3.3-70b-versatile",
#         temperature=0
#     )


# @st.cache_resource
# def load_embeddings() -> OpenAIEmbeddings:
#     """
#     Returns a cached OpenAIEmbeddings instance.

#     text-embedding-3-small:
#       - 1536-dimensional output vectors
#       - Much cheaper than text-embedding-ada-002
#       - Better quality than ada-002 on most benchmarks
#     """
#     return OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         openai_api_key=os.getenv("OPENAI_API_KEY")
#     )


import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )


@st.cache_resource
def load_embeddings():
    # 100% free — runs locally, no API key needed
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
