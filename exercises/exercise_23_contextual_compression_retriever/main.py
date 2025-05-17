from llm.openai import LLM
import streamlit as st
from utils.file_utils import FileUtils
from pathlib import Path
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from utils.chroma_db import ChromaDB

def run():
    ContextualCompressionRetrieverDemo()

class ContextualCompressionRetrieverDemo:
    def __init__(self):
        self.llm = LLM().openai_gpt4o()
        contents = FileUtils.contents(
            file_name="contextual_compression_retriever", 
            ext="md",
            base_dir=Path(__file__).parent
        )
        st.markdown(contents)
        st.title("Exercise")
        # simple cosine distance based retriever
        similarity_retriever = ChromaDB.from_docs().as_retriever(search_type="similarity",
                                              search_kwargs={"k": 3})

        # retrieves the documents similar to query and then applies the compressor
        compression_retriever_chain_extractor = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(llm=self.llm),
            base_retriever=similarity_retriever
        )
        compression_retriever_chain_filter = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm=self.llm), 
            base_retriever=similarity_retriever
        )

        user_query = st.text_input("Enter your query")
        if st.button("Retrieve - LLMChainExtractor") and user_query.strip():
            docs = compression_retriever_chain_extractor.invoke(user_query)
            st.write(docs)
        
        if st.button("Retrieve - LLMChainFilter") and user_query.strip():
            docs = compression_retriever_chain_filter.invoke(user_query)
            st.write(docs)


        
        