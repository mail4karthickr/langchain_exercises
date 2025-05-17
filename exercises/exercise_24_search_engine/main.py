from functools import cached_property
from importlib import metadata
import io
import json
import logging
from exercises.exercise_23_contextual_compression_retriever.main import ContextualCompressionRetrieverDemo
import streamlit as st
import os
from langchain.document_loaders import JSONLoader
from langchain.docstore.document import Document
from llm.openai import LLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from glob import glob
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def run():
    SearchEngine()

class SearchEngine:
    def __init__(self):
        self.llm = LLM().openai_gpt4o()
        self.wiki_docs = SearchEngine.wiki_data()

        # laod pdf
        # self.paper_docs = []
        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
        if 'data_ingested' not in st.session_state:
            st.session_state.data_ingested = False

        if st.button('Ingest data'):
            chroma_db = self.data_ingest_or_load()
            st.session_state.chroma_db = chroma_db
            st.session_state.data_ingested = True

        if st.session_state.data_ingested:
            user_input = st.text_input("ask")
            similarity_retriever =  st.session_state.chroma_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            if st.button("Similarity Retriever") and user_input.strip():
                top_docs = similarity_retriever.invoke(user_input)
                self.display_docs(top_docs)
            if st.button("Multi Query Retriever") and user_input.strip():
                self.set_logging()
                mq_retriever = MultiQueryRetriever.from_llm(
                    retriever = similarity_retriever,
                    llm = self.llm
                )
                top_docs = mq_retriever.invoke(user_input)
                self.display_logs()
                self.display_docs(top_docs)
            if st.button("Contextual Compression Retrieval - LLMChainExtractor") and user_input.strip():
                self.set_logging()
                mq_retriever = MultiQueryRetriever.from_llm(
                    retriever = similarity_retriever,
                    llm = self.llm
                )
                compressor = LLMChainExtractor.from_llm(llm=self.llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=mq_retriever
                )
                top_docs = compression_retriever.invoke(user_input)
                self.display_logs()
                self.display_docs(top_docs)
            if st.button("Contextual Compression Retrieval - LLMChainFilter") and user_input.strip():
                self.set_logging()
                mq_retriever = MultiQueryRetriever.from_llm(
                    retriever = similarity_retriever,
                    llm = self.llm
                )
                compressor = LLMChainFilter.from_llm(llm=self.llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=mq_retriever
                )
                top_docs = compression_retriever.invoke(user_input)
                self.display_logs()
                self.display_docs(top_docs)
            if st.button("Chained Retrieval Pipeline") and user_input.strip():
                # Similarity Retrieval → Compression Filter → Reranker Model Retrieval
                _filter = LLMChainFilter.from_llm(llm=self.llm)
                # Retriever 2 - retrieves the documents similar to query and then applies the filter
                compressor_retriever = ContextualCompressionRetriever(
                    base_compressor=_filter, base_retriever=similarity_retriever
                )
                # download an open-source reranker model - BAAI/bge-reranker-v2-m3
                reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
                reranker_compressor = CrossEncoderReranker(model=reranker, top_n=3)
                # Retriever 3 - Uses a Reranker model to rerank retrieval results from the previous retriever
                final_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker_compressor, base_retriever=compressor_retriever
                )
                top_docs = final_retriever.invoke(user_input)
                self.display_docs(top_docs)
    
    def set_logging(self):
        self.log_buffer = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setLevel(logging.INFO)
        
        # Attach only to MultiQueryRetriever logger
        self.logger = logging.getLogger("langchain.retrievers.multi_query")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)

    def display_logs(self):
        self.log_handler.flush()
        log_output = self.log_buffer.getvalue()
        st.subheader("MultiQuery Logs")
        st.code(log_output, language="text")

        # Optional: reset buffer if needed
        self.log_buffer.truncate(0)
        self.log_buffer.seek(0)

    def display_docs(self, docs):
        for doc in docs:
            st.write('Metadata:', doc.metadata)
            st.write('Content Brief:')
            st.markdown(doc.page_content[:1000])


    def data_ingest_or_load(self):
        ingestion_flag_file = "./.ingestion_done"
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "rag_docs")
        pdf_files = glob(f'{file_path}/*.pdf')
        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

        # If ingestion done load chroma from disk
        if os.path.exists(ingestion_flag_file):
            st.success("✅ Data ingestion already completed.")
            return Chroma(
                persist_directory="./my_db",
                collection_name='my_db',
                embedding_function=openai_embed_model
            )
        
        # Else: Perform ingestion 
        st.warning("⏳ Data ingestion in progress...")
        paper_docs = []
        for file_path in pdf_files:
            st.write(f"Creating contextual chunk for {file_path}")
            contextual_chunks = self.create_contextual_chunks(file_path=file_path)
            st.write(f"Finished contextual chunk for {file_path}")
            paper_docs.extend(contextual_chunks)
        
        total_docs = self.wiki_docs + paper_docs
        chroma = Chroma.from_documents(
            documents=total_docs,
            collection_name='my_db',
            embedding=openai_embed_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory="./my_db"
        )

        # Make ingestion complete
        with open(ingestion_flag_file, "w") as file:
            file.write("done")
        
        st.success("✅ Ingestion completed and DB created.")
        return chroma
            

    @staticmethod
    def wiki_data():
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "rag_docs", "wikidata_rag_demo.jsonl")
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".",
            text_content=False,
            json_lines=True
        )
        wiki_docs = loader.load()
        wiki_docs_processed = []
        for doc in wiki_docs:
            doc = json.loads(doc.page_content)
            metadata = {
                "id": doc["id"],
                "title": doc["title"],
                "source": "Wikipedia"
            }
            data = ' '.join(doc["paragraphs"])
            wiki_docs_processed.append(Document(page_content=data, metadata=metadata))
        return wiki_docs_processed
    
    def generate_chunk_context(self, document, chunk):
        chunk_process_prompt = """
        Act as a reserach paper analysis,
        Your task is to provide breif, relevant context for a chunk of text
        based on the following research paper.

        Here is the research paper:
        <paper>
        {paper}
        </paper>

        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>

        - Give a short succinct context to situate this chunk within the overall document
            for the purposes of improving search retrieval of the chunk.
        - Answer only with the succinct context and nothing else.
        - Context should be mentioned like 'Focuses on ....'
        do not mention 'this chunk or section focuses on...'

        Context:
        """

        prompt_template = ChatPromptTemplate.from_template(chunk_process_prompt)
        chunk_chain = prompt_template | self.llm | StrOutputParser()
        context = chunk_chain.invoke({
            'paper': document,
            'chunk': chunk
        })
        return context
    
    def create_contextual_chunks(self, file_path):
        loader = PyMuPDFLoader(file_path)
        doc_pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
        doc_chunks = splitter.split_documents(doc_pages)
        
        # Generating contextual chunks
        contextual_chunks = []
        original_doc = '\n'.join([doc.page_content for doc in doc_pages])
        for chunk in doc_chunks:
            context = self.generate_chunk_context(document=original_doc, chunk=chunk)
            contextual_chunks.append(
                Document(
                    page_content=context+"\n"+chunk.page_content, 
                    metadata=chunk.metadata
                )
            )
        return contextual_chunks