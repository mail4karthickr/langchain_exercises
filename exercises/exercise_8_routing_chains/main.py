import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_groq import ChatGroq
from llm.openai import LLM
from langchain_core.runnables import RunnableBranch

def run():
    RoutingChains().run()

class RoutingChains:
    def __init__(self):
        self.model = LLM().chat_groq
    
    def run(self):
        st.markdown(self.project)
        review_input = st.text_area("Enter customer review:", height=200)
        instruction = st.text_input("Instructions")
        if st.button("Analyze review"):
            if review_input.strip() == "" or instruction.strip() == "":
                st.warning("Please enter a review before clicking the button.")
            else:
                branch = RunnableBranch(
                    (lambda x: "summarize" in x["topic"].lower(), self.summary_chain()),
                    (lambda x: "sentiment" in x["topic"].lower(), self.sentiment_chain()),
                    (lambda x: "email" in x["topic"].lower(), self.email_chain()),
                    self.default_answer,
                )
                full_chain = ({
                    "topic": self.classifier_chain(),
                    "instruction": lambda input_prompt: input_prompt.get("instruction"),
                    "review": lambda input_prompt: input_prompt.get("review"),
                    "sentiment": lambda input_prompt: self.sentiment_chain()

                } | branch)
                response = full_chain.invoke({'review': review_input, 'instruction': instruction})
                st.write(response)

    def default_answer(query):
        return "Sorry instructions are not the defined intents"

    @property
    def project(self):
        return """
            ## Routing Chains with LCEL

            The idea here is to have multiple individual LLM Chains which can perform their own tasks like summarize, sentiment etc.

            We also have a router chain which can classify the user prompt intent and then route the user prompt to the relevant LLM Chain e.g if the user wants to summarize an article, his prompt request would be routed to the summarize chain automatically to get the result
        """
    
    def classifier_chain(self):
        prompt = """
            Given the user instructions below for analyzing customer review,
            classify it as only one of the following categories:
                - summarize
                - sentiment
                - email

            Do not respond with more than one word.

            Instructions:
            {instruction}
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        return prompt_template | self.model | StrOutputParser()
    
    def summary_chain(self):
        prompt = """
            Act as a customer review analyst, given the following customer review,
            generate a short summary (max 2 lines) of the review.

            Customer Review:
            {review}
        """
        return self.chain(prompt)
    
    def sentiment_chain(self):
        prompt = """
            Act as a customer review analyst, given the following customer review,
            find out the sentiment of the review.
            The sentiment can be either positive, negative or neutral.
            Return the result as a single word.

            Customer Review:
            {review}
        """
        return self.chain(prompt)
    
    def email_chain(self):
        prompt = """
            Act as a customer review analyst,
            given the following customer review and its sentiment
            generate an email response to the customer based on the following conditions.
                - If the sentiment is positive or neutral thank them for their review
                - If the sentiment is negative, apologize to them

            Customer Review:
            {review}
            Sentiment:
            {sentiment}
        """
        return self.chain(prompt)

    def chain(self, prompt):
        prompt_template = ChatPromptTemplate.from_template(prompt)
        return prompt_template | self.model | StrOutputParser()