import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_groq import ChatGroq
from llm.openai import LLM

def run():
    BranchingAndMergingChain().run()

class BranchingAndMergingChain:
    def __init__(self):
        self.model = LLM().chat_groq
        
    def run(self):
        st.markdown(self.project())
        topic = st.text_input("Enter a topic (e.g. Generative AI):", value="Generative AI")
        if st.button("Generate Report"):
            branch_chain = RunnableParallel(
                topic=itemgetter('topic'),
                description=self.description_chain,
                pros=self.pros_chain,
                cons=self.cons_chain
            )
            final_chain = branch_chain | self.create_report()
            response = final_chain.invoke({"topic": topic})
            st.write(response)

    def project(self):
        return """
        ## Branching and Merging Chains with LCEL
        The idea here is to have multiple branching LLM Chains which work independently in parallel and then we merge their outputs finally using a merge LLM chain at the end to get a consolidated output
        """

    def description_chain(self, topic: str):
        prompt = """
            Generate a two line description for the topic delimited with backticks:
            ```{topic}```
        """
        return self.chain_invoke(prompt, topic)
    
    def pros_chain(self, topic: str):
         prompt = """
            Generate three bullet points talking about the pros for the given topic:
            ```{topic}```
        """
         return self.chain_invoke(prompt, topic)

    def cons_chain(self, topic: str):
         prompt = """
            Generate three bullet points talking about the cons for the given topic:
            ```{topic}```
        """
         return self.chain_invoke(prompt, topic)
    
    def chain_invoke(self, prompt: str, topic: str):
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | self.model | StrOutputParser()
        return chain.invoke({'topic': topic})
    
    def create_report(self):
        merge_prompt = """
            Create a report about {topic} with the following information:
            Description:
            {description}
            Pros:
            {pros}
            Cons:
            {cons}

            Report should be in the following format:

            Topic: <name of the topic>

            Description: <description of the topic>

            Pros and Cons:

            <table with two columns showing the 3 pros and cons of the topic>     
        """
        prompt_template = ChatPromptTemplate.from_template(merge_prompt)
        return prompt_template | self.model | StrOutputParser()
    

    
    