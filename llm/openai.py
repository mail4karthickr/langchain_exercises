from functools import cached_property
from langchain_openai import ChatOpenAI
import streamlit as st

class LLM:
    @cached_property
    def openai(self):
        return ChatOpenAI(
            model_name="gpt-4",
            temperature=0.0,
            openai_api_key=st.session_state.openai_api_key
        )