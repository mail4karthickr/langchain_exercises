import os
import streamlit as st
from llm.openai import LLM
from langchain.retrievers.multi_query import MultiQueryRetriever 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from utils.file_utils import FileUtils
from pathlib import Path
import logging
import io

def run():
    MultiQueryRetrieval()

class MultiQueryRetrieval:
    def __init__(self):
        self.log_buffer = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setLevel(logging.INFO)
        
        # Attach only to MultiQueryRetriever logger
        self.logger = logging.getLogger("langchain.retrievers.multi_query")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)

        # LLM Setup
        self.llm = LLM().openai_gpt4o(temperature=0.0)

        # UI
        st.markdown(self.multi_query_desc())
        st.title("Example")

        user_query = st.text_input("Enter your query")
        if st.button("Retrieve") and user_query.strip():
            similarity_retriever = self.chroma_db().as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            mq_retriever = MultiQueryRetriever.from_llm(
                retriever=similarity_retriever,
                llm=self.llm
            )
            docs = mq_retriever.invoke(user_query)
            st.subheader("Retrieved Documents")
            st.write(docs)
        
            # Flush and display logs
            self.log_handler.flush()
            log_output = self.log_buffer.getvalue()
            st.subheader("MultiQuery Logs")
            st.code(log_output, language="text")

             # Optional: reset buffer if needed
            self.log_buffer.truncate(0)
            self.log_buffer.seek(0)

    def chroma_db(self):
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
        return Chroma.from_texts(
            texts=self.docs(), 
            collection_name='db_docs',
            # need to set the distance function to cosine else it uses euclidean by default
            # check https://docs.trychroma.com/guides#changing-the-distance-function
            collection_metadata={"hnsw:space": "cosine"},
            embedding=openai_embed_model
        )

    def docs(self):
        return [
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

    def multi_query_desc(self):
        return FileUtils.contents(
            file_name="multi_query_retrieval", 
            ext="md",
            base_dir=Path(__file__).parent
        )
