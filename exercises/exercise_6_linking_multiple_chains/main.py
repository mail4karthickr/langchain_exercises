from llm.openai import LLM
import streamlit as st
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

def run():
    LinkingMultipleChains().run()

class LinkingMultipleChains:
    def __init__(self):
        self.model = LLM().chat_groq

    def project(self):
        return """
        ## Linking Multiple Chains Sequentially in LCEL

        Here we will see how we can link several LLM Chains sequentially using LCEL.

        Typically the output from one chain might go as input into the next chain and so on.

        The overall chain would run each chain sequentially in order till we get the final output which can be a combination of intermediate outputs and inputs from the previous chains.
        """

    def run(self):
        st.markdown(self.project())
        user_input = st.text_area(
            label="Enter support tickets (comma-separated)",
            value=self.default_text,
            height=200
        )
        if st.button("Analyze Tickets"):
            tickets = [ticket.strip() for ticket in user_input.split(",") if ticket.strip()]
            final_chain = self.detect_language() | self.translate_to_eng() | self.resolution_response_in_eng() | self.translate_response_to_src_lang()
            response = final_chain.map().invoke([{'orig_msg': ticket} for ticket in tickets])
            st.dataframe(pd.DataFrame(response), use_container_width=True)

    # chain 1
    # Detect the language of the customer ticket
    def detect_language(self) -> RunnablePassthrough:
        prompt = """
            Act as a customer support agent.
            For the customer support message delimited below by triple brackets,
            Output the language of the message in one word only, e.g. Spanish

            Customer Message:
            ```{orig_msg}```
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | self.model | StrOutputParser()
        return RunnablePassthrough.assign(orig_lang=chain)
    
    # chain 2
    #  Give the ticket and the source language translate it to english
    def translate_to_eng(self) -> RunnablePassthrough:
        prompt = """
            Act as a customer support agent.
            For the customer message and customer message language delimited below by triple backticks,
            Translate the customer message from the customer message language to English
            if customer message language is not in English,
            else return back the original customer message.
            Return **only** the translated message. Do not add any explanations or extra text.

            Customer Message:
                ```{orig_msg}```
            Customer Message Language:
                ```{orig_lang}```
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | self.model | StrOutputParser()
        return RunnablePassthrough.assign(trans_msg=chain)

    # chain 3
    # Given the response in english
    def resolution_response_in_eng(self) -> RunnablePassthrough:
        prompt = """
            Act as a customer support agent.
            For the customer support message delimited by triple backticks,
            Generate an appropriate resolution response in English.
            Return **only** the response. Do not add any explanations or extra text.

            Customer Message:
            ```{trans_msg}```
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | self.model | StrOutputParser()
        return RunnablePassthrough.assign(trans_response=chain)
    
    # chain 4
    # Given the response in english
    def translate_response_to_src_lang(self) -> RunnablePassthrough:
        prompt = """
            Act as a customer support agent.
            For the customer resolution response and target language delimited below by triple backticks,
            Translate the customer resolution response message from English to the target language
            if target language is not in English,
            else return back the original customer resolution response.
            Do not include any preamble, explanation, or formatting.

            Customer Resolution Response:
            ```{trans_response}```
            Target Language:
            ```{orig_lang}```
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | self.model | StrOutputParser()
        return RunnablePassthrough.assign(orig_response=chain)
    
    @property
    def default_text(self):
        return """
        "I can't access my email. It keeps showing an error message. Please help.",
        "Tengo problemas con la VPN. No puedo conectarme a la red de la empresa. ¿Pueden ayudarme, por favor?",
        "Mon imprimante ne répond pas et n'imprime plus. J'ai besoin d'aide pour la réparer.",
        "我无法访问公司的网站。每次都显示错误信息。请帮忙解决。"
        """