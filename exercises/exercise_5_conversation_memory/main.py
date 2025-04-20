import time
from typing import List, Tuple
import sys
from llm.openai import LLM
import streamlit as st
from colorama import Fore, Style, init
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_groq import ChatGroq

# class ConversationMemory:
#     def run(self):
#         st.write("Links")
#         st.write("https://www.analyticsvidhya.com/blog/2024/11/langchain-memory/")
#         st.write("https://python.langchain.com/docs/how_to/chatbots_memory/")
#         pass


def run():
    ChatBot().run()

class ChatBot:
    def __init__(self):
        self.model = LLM().chat_groq
    
    def run(self):
        # Setup memory in session state
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(k=50, return_messages=True)

        # Define prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Define chain
        chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(st.session_state.memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | self.model
        )

        # Streamlit UI
        st.title("🤖 Groq ChatBot")
        st.markdown("Ask anything! Powered by `llama3-8b-8192` via Groq.")


        # Display chat history from memory
        for msg in st.session_state.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant").write(msg.content)

        # Handle user input
        user_input = st.chat_input("Type your message here...")

        if user_input:
            st.chat_message("user").write(user_input)

            start = time.time()
            response = chain.invoke({"input": user_input})
            end = time.time()

            # Save context to memory
            st.session_state.memory.save_context({"input": user_input}, {"output": response.content})

            # Show AI response
            st.chat_message("assistant").write(response.content)
            st.caption(f"⏱️ Response time: {end - start:.2f}s")