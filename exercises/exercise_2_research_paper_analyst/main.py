from functools import cached_property
import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

def run():
    if "analyst" not in st.session_state:
        st.session_state.analyst = ResearchPapaerAnalyst()
    st.session_state.analyst.run()


class ResearchPapaerAnalyst:
    def __init__(self):
        self.messages = []
        self.chatgpt = ChatOpenAI(
            model_name="gpt-4", 
            temperature=0.0,
            openai_api_key=st.session_state.openai_api_key
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.sys_prompt),
            ("human", "{instruction}")
        ])

    def run(self):
        st.markdown(self.project)
        if st.button('General Audience Summary'):
            self.general_audience_summary()
        if st.button("Health Care Company Report"):
            self.healthcare_report()
        if st.button("GenAI Company Healthcare Report"):
            self.genai_healthcare_report()

    @property
    def project(self):
        return """
        ## Mini-Project 2: Research Paper Analyst
        
        Make ChatGPT act as an AI expert and transform the given research paper abstract based on the nature of the audience mentioned below.

        - Short summary of maximum 10 lines for a general audience
        - Detailed report for a healthcare company. Have bullet points for pros and cons of ethics in Generative AI as mentioned in the paper
        - Detailed report for a generative AI company solving healthcare problems. Have bullet points for key points mentioned for Generative AI for text, images and structured data based healthcare

        Try to use `ChatPromptTemplate` so you can have a conversation with ChatGPT for each of the above tasks using conversational prompting
        """
    
    @property
    def sys_prompt(self):
        return """
            Act as a Artificial Intelligence Expert.
            Transform the input research paper abstract given below
            based on the instruction input by the user.
        """
    
    @cached_property
    def abstract(self) -> str:
        abstract_path = os.path.join(os.path.dirname(__file__), "paper_abstract.txt")
        with open(abstract_path, "r") as f:
            return f.read().strip()
    
    def general_audience_summary(self):
        prompt_txt = f"""
            Based on the following research paper abstract,
            create the summary report of maximum 10 lines
            for a general audience

            Abstract:
            {self.abstract}
        """
        self.send_message(prompt_txt)

    def healthcare_report(self):
        prompt_txt = f"""
            Use only the research paper abstract from earlier and create a detailed report for a healthcare compnay.
            In the report include max 3 bullet points for pros and cons of ethics in Generative AI.
            If abstract for the paper is not found in the earlier chat then politely respond saying abstract not found.
        """
        self.send_message(prompt_txt)

    def genai_healthcare_report(self):
        prompt_txt = f"""
            Use only the research paper abstract from earlier and create a detailed report for a generative AI company solving healthcare problems.
            In the report also include sections for key points mentioned around Generative AI for text, images and structured data based healthcare
        """
        self.send_message(prompt_txt)
    
    def send_message(self, prompt_txt):
        self.messages.append(HumanMessage(content=prompt_txt))
        chain = self.prompt | self.chatgpt
        response = chain.invoke({"instruction": self.messages})
        self.messages.append(response)
        st.write(response.content)
        st.write(len(self.messages))

