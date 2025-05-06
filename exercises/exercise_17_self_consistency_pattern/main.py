import streamlit as st
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from llm.openai import LLM

def run():
    SelfConsistencyPattern().run()

class SelfConsistencyPattern:
    def run(self):
        st.markdown(self.definition())
        st.title("🧠 Swift Code Assistant with Chain-of-Thought")
        question = st.text_area("Enter your Swift Programming question")
        
        if st.button("Generate Answer") and question.strip():
            gpt = LLM().openai_gpt4o(temperature=0)
            gpt1 = LLM().openai_gpt4o(temperature=0)
            gpt2 = LLM().openai_gpt4o(temperature=0.5)
            gpt3 = LLM().openai_gpt4o(temperature=0.9)

            chain1 = self.cot_prompt_template() | gpt1 | StrOutputParser()
            chain2 = self.cot_prompt_template() | gpt2 | StrOutputParser()
            chain3 = self.cot_prompt_template() | gpt3 | StrOutputParser()

            sc_chain = (
                RunnableParallel(
                    problem=RunnablePassthrough(),
                    reasoning_path_1=chain1,
                    reasoning_path_2=chain2,
                    reasoning_path_3=chain3
                )
                | self.self_con_prompt_template()
                | gpt
            )
            response = sc_chain.invoke({'problem': question})
            st.write(response.content)
    
    def cot_prompt_template(self):
        return ChatPromptTemplate.from_template("""
            You are a senior Swift developer assisting a junior developer.

            Your responsibilities:
            - ✅ Accept only **Swift programming questions**
            - ❌ If the question is not related to Swift, politely respond:
            "I'm here to assist only with Swift programming questions. Please provide a Swift-related question."

            If the question **is related to Swift**, think through the problem **step-by-step** and follow this format:

            1. **Understand the problem**: Briefly explain what the question is asking.
            2. **Plan the solution**: Outline the logic or algorithm you will use.
            3. **Write the Swift code**: Provide the full, correct Swift function.
            4. **(Optional)** Add a short test example to demonstrate how the function works.
            5. **Important**: End your response with ONLY the Swift code. ❌ Do not include markdown (like ```swift) or extra commentary.

            Problem:
            {problem}
        """)
    
    def self_con_prompt_template(self):
        return ChatPromptTemplate.from_template("""
            Given the following problem
            and 3 diverse reasoning paths explored by an AI model
            analyse these pathways carefully,
            aggregate, take the majority vote as needed
            and generate a final single reasoning path along with the answer

            AI Model Reasoning Path 1:
            {reasoning_path_1}

            AI Model Reasoning Path 2:
            {reasoning_path_2}

            AI Model Reasoning Path 3:
            {reasoning_path_3}

            Problem:
            {problem}
            """
        )
    
    def definition(self):
        return """
        # 🧠 Self-Consistency Pattern in Prompt Engineering

        ## 📖 Definition
        The **Self-Consistency Pattern** is a prompt engineering technique where a language model is asked to solve the same problem multiple times (with randomness enabled), and the most frequently occurring answer is selected as the final output. This helps overcome occasional errors in reasoning by leveraging statistical consensus among multiple completions.

        It is especially effective when used with **Chain-of-Thought (CoT)** prompting, where the model is encouraged to reason step-by-step.
        """

