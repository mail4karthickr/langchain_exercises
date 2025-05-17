from langchain_chroma import Chroma
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings

class ChromaDB:
    @staticmethod
    def from_docs():
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
        docs = [
            'Quantum mechanics describes the behavior of very small particles.',
            'Photosynthesis is the process by which green plants make food using sunlight.',
            'Artificial Intelligence aims to create machines that can think and learn.',
            'The pyramids of Egypt are historical monuments that have stood for thousands of years.',
            'New Delhi is the capital of India and the seat of all three branches of the Government of India.',
            'Biology is the study of living organisms and their interactions with the environment.',
            'Music therapy can aid in the mental well-being of individuals.',
            'Mumbai is the financial capital and the most populous city of India. It is the financial, commercial, and entertainment capital of South Asia.',
            'The Milky Way is just one of billions of galaxies in the universe.',
            'Economic theories help understand the distribution of resources in society.',
            'Kolkata is the de facto cultural capital of India and a historically and culturally significant city. Calcutta served as the de facto capital of India until 1911.',
            'Yoga is an ancient practice that involves physical postures and meditation.'
        ]
        # create vector DB of docs and embeddings - takes 1 min on Colab
        return Chroma.from_texts(
            texts=docs, 
            collection_name='db_docs',
            # need to set the distance function to cosine else it uses euclidean by default
            # check https://docs.trychroma.com/guides#changing-the-distance-function
            collection_metadata={"hnsw:space": "cosine"},
            embedding=openai_embed_model
        )