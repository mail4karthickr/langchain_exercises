from functools import cached_property
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_groq import ChatGroq

class LLM:
    @cached_property
    def openai(self):
        return ChatOpenAI(
            model_name="gpt-4",
            temperature=0.0,
            openai_api_key=st.session_state.openai_api_key
        )
    
    @cached_property
    def chat_groq(self):
        return ChatGroq(
            api_key=st.session_state.groq_api_key,
            model_name="llama3-8b-8192"
        )
