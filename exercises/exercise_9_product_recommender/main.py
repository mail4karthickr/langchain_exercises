from operator import itemgetter
import re
from llm.openai import LLM
import streamlit as st
import pandas as pd
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pathlib import Path
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableMap

def run():
    ProductRecommender().run()

class ProductRecommender:
    def __init__(self):
        self.model = LLM().chat_groq
        current_dir = Path(__file__).parent
        csv_path = current_dir / "Ecommerce_Product_List.csv"
        self.df = pd.read_csv(csv_path)
        
    def run(self):
        user_name = st.text_input('Username')
        user_query = st.chat_input("Ask something...")
        if user_name and user_query:
            self.chat_with_llm(prompt=user_query, session_id=user_name)

    def chat_with_llm(self, prompt: str, session_id: str):
        combined_chain = (
            RunnableLambda(lambda x: {
                "user_query": x["human_input"],
                "history": x["history"]
            })
            | RunnablePassthrough.assign(
                history=lambda x: self.memory_buffer_window(x["history"]),
                human_input=lambda x: x["user_query"]
            )
            | self.repharse_query_chain()
            | RunnableLambda(lambda query: {"user_query": query})
            | self.text_to_pandas_chain()
            | RunnableLambda(lambda table: {
                "product_table": table,
                "user_query": prompt
            })
            | self.product_description_chain()
        )

        conv_chain = RunnableWithMessageHistory(
            combined_chain,
            self.get_sesssion_history_db,
            input_messages_key="human_input",
            history_messages_key="history",
        )

        response = conv_chain.invoke(
            {"human_input": prompt},
            config={'configurable': {'session_id': session_id}}
        )

        st.chat_message(name="system").markdown(response)

    def get_sesssion_history_db(self, session_id):
        return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

    def repharse_query_chain(self):
        sys_prompt = """You are a retail product expert.
                Carefully analyze the following conversation history
                and the current user query.
                Refer to the history and rephrase the current user query
                into a standalone query which can be used without the history
                for making search queries.
                Rephrase only if needed.
                Just return the query and do not answer it.
            """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", """Current User Query: {human_input}""")
        ])
        return (
            RunnablePassthrough.assign(history=lambda x: self.memory_buffer_window(x["history"]))
            | prompt_template
            | self.model 
            | StrOutputParser()
        )

    # create a memory buffer window function to return the last K conversations
    def memory_buffer_window(self, messages, k=10): # 10 here means retrieve only last 2*10 user-AI conversations
        return messages[-(2*k):]

    def execute_pandas_query(self, query):
        st.write("Running query:", query)
        try:
            # clean_query = self.sanitize_query(query)
            result_df = eval(query)  # Safe context
            return result_df.to_markdown()
        except Exception as e:
            return f"❌ Error evaluating query:\n```{e}```"


    def text_to_pandas_chain(self):
        FILTER_PROMPT = """
            ⚠️ STRICT: Return ONLY the pandas query using `self.df`. No explanation. No markdown. Your output MUST start with `self.df`.

            Given the following schema of a dataframe table,
            your task is to figure out the best pandas query to
            filter the dataframe based on the user query which
            will be in natural language.

            The schema is as follows:

            #   Column        Non-Null Count  Dtype
            ---  ------        --------------  -----
            0   Product_ID    30 non-null     object
            1   Product_Name  30 non-null     object
            2   Category      30 non-null     object
            3   Price_USD     30 non-null     int64
            4   Rating        30 non-null     float64
            5   Description   30 non-null     object

            Category has values: ['Laptop', 'Tablet', 'Smartphone',
                                  'Smartwatch', 'Camera',
                                  'Headphones', 'Mouse', 'Keyboard',
                                  'Monitor', 'Charger']

            Rating ranges from 1 - 5 in floats

            You will try to figure out the pandas query focusing
            only on Category, Price_USD and Rating if the user mentions
            anything about these in their natural language query.
            Do not display any rows which dose not meet the conditions provided in the below user query.
            Do not make up column names, only use the above.
            If not the pandas query should just return the full dataframe.
          
            User Query: {user_query}
        """
        filter_prompt_template = ChatPromptTemplate.from_template(FILTER_PROMPT)
        return (
            filter_prompt_template 
            | self.model 
            | StrOutputParser()
            | RunnableLambda(self.debug_output)
            | RunnableLambda(lambda query: self.execute_pandas_query(query))
        )
    
    def product_description_chain(self):
        RECOMMEND_PROMPT = """
            Act as an expert retail product advisor
            Given the following table of products,
            focus on the product attributes and description in the table
            and based on the user query below do the following

            - Recommend the most appropriate products based on the query
            - Recommedation should have product name, price,  rating, description
            - Also add a brief on why you recommend the product
            - Do not make up products or recommend products not in the table
            - If some specifications do not match focus on the ones which match and recommend
            - If nothing matches recommend 5 random products from the table
            - Do not generate anything else except the fields mentioned above

            In case the user query is just a generic query or greeting
            respond to them appropriately without recommending any products

            Product Table:
            {product_table}

            User Query:
            {user_query}

            Recommendation:
        """
        recommend_prompt_template = ChatPromptTemplate.from_template(RECOMMEND_PROMPT)
        return (
            recommend_prompt_template
            | self.model
            | StrOutputParser()
        )
    
    @staticmethod
    def debug_output(output):
        st.text_area("🔍 Output from StrOutputParser:", value=output, height=150)
        return output

