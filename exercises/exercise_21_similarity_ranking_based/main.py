import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def run():
    SimilarityAndRankingBased()

class SimilarityAndRankingBased:
    def __init__(self):
        st.title("Similarity and Ranking Based")
        st.write("We use cosine similarity here and retrieve the top 3 similar documents based on the user input query")
        st.markdown(self.md_contents())
        user_query = st.text_input("Enter your query")
        if st.button("Similarity Retriever"):
            if user_query.strip():
                similarity_retriever = self.chroma_db(docs=self.docs()).as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                top3_docs = similarity_retriever.invoke(user_query)
                st.write(top3_docs)

        if st.button("MMR search"):
            if user_query.strip():
                mmr_retriever = self.chroma_db(docs=self.docs()).as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        'k': 3,
                        'fetch_k': 10
                    }
                )
                top3_docs = mmr_retriever.invoke(user_query)
                st.write(top3_docs)

        st.markdown(
            """
            ## 🎯 `similarity_score_threshold` in LangChain

            ### 🔍 What is it?

            `similarity_score_threshold` is a **document retrieval strategy** used in vector databases (like Chroma) that **filters documents based on a minimum similarity score** instead of just retrieving the top `k`.

            ---

            ### 🧠 Concept

            Instead of always returning the top `k` most similar documents (regardless of how relevant they are), this method:

            - Computes similarity (usually cosine similarity) between the query and documents.
            - Returns **only those documents whose score is above the specified threshold**.

            ---

            ### ✅ When to Use It

            Use `similarity_score_threshold` when:

            - You want to **avoid irrelevant or low-quality results**.
            - You're okay with getting **fewer than `k` documents**.
            - You want **high-precision filtering**, e.g., for compliance, summarization, or exact search use cases.

            ---

            ### 🧪 Example in LangChain

            ```python
            retriever = chroma_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.75  # Only return documents with cosine similarity ≥ 0.75
                }
            )
            """
        )
        if st.button("Similarity with threshold"):
            if user_query.strip():
                similarity_threshold_retriever = self.chroma_db(docs=self.docs()).as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        'k': 3,
                        "score_threshold": 0.3
                    }
                )
                top3_docs = similarity_threshold_retriever.invoke(user_query)
                st.write(top3_docs)
        self.custom_retriever_desc()
                

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
    
    def md_contents(self):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "similarity_vs_mmr_retriever.md")

        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
            return md_content
        
    def chroma_db(self, docs):
        # details here: https://openai.com/blog/new-embedding-models-and-api-updates
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

        # create vector DB of docs and embeddings
        chroma_db = Chroma.from_texts(
            texts=docs,
            collection_name='db_docs',
            # need to set the distance function to cosine else it uses euclidean by default
            # check https://docs.trychroma.com/guides#changing-the-distance-function
            collection_metadata={"hnsw:space": "cosine"},
            embedding=openai_embed_model
        )
        return chroma_db

    def custom_retriever_desc(self):
        st.markdown(
            """
            ### Custom Retriever with Similarity Scores + Thresholding

            Here we will create a custom retriever which will:

            - Retrieve documents with cosine distance
            - Convert to similarity score and apply thresholding
            - Return topk documents above a similarity threshold
            """
        )
    
    def multi_query_retriever(self):
        st.markdown()

