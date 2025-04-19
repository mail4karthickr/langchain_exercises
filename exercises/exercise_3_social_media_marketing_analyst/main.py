import streamlit as st
from utils.file_utils import FileUtils
from llm.openai import LLM
from langchain_core.prompts import ChatPromptTemplate

def run():
    SocialMediaMarketingAnalyst().run()

class SocialMediaMarketingAnalyst:
    def __init__(self):
        self.openai = LLM().openai
        
    def run(self):
        st.markdown(self.project())
        if st.button("📝 Generate Basic Product Description"):
            self.product_description()
        if st.button("📋 Generate Detailed Product Description"):
            self.formatted_product_description()
        if st.button("✨ Catchy Description with Emojis"):
            self.catchy_product_description()

    def product_description(self) -> str:
        prompt_txt = """
            Act as a marketing manager.
            Your task is to help a marketing team create a
            description for a retail website advert of a product based
            on a technical fact sheet specifications for a mobile smartphone
            ​
            Write a brief product description

            Technical specifications:
            {fact_sheet_mobile}
        """
        self.request_openai(prompt_txt)
      
    def formatted_product_description(self):
        prompt_txt = """
            Act as a marketing manager.
            Your task is to help a marketing team create a
            description for a retail website advert of a product based
            on a technical fact sheet specifications for a mobile smartphone
            ​
            The description should follow this format:

            Product Name: <Name of the smartphone>
            ​
            Description: <Brief Overview of the features>
            ​
            Product Specifications:
            <Table with key product feature specifications>
            ​
            The description should focus on the most important features
            a customer might look for in a phone including the foldable display screen, processing power, RAM, camera and battery life.
            ​
            After the description, the table should have the
            key specifications of the product. It should have two columns.
            The first column should have 'Feature'
            and the second column should have 'Specification'
            and try to put exact numeric values for features if they exist.
            Only put these features in the table - foldable display screen, processing power, RAM, camera and battery life

            Technical specifications:
            {fact_sheet_mobile}
        """
        self.request_openai(prompt_txt)
    
    def catchy_product_description(self):
        prompt_txt = """
            Act as a marketing manager.
            Your task is to help a marketing team create a
            description for a retail website advert of a product based
            on a technical fact sheet specifications for a mobile smartphone
            ​
            Write a catchy product description with some emojis,
            which uses at most 60 words
            and focuses on the most important things about the smartphone
            which might matter to users like display and camera

            Technical specifications:
            {fact_sheet_mobile}
        """
        self.request_openai(prompt_txt)

    def project(self) -> str:
        return """
            ## Mini-Project 3: Social Media Marketing Analyst

            You have the technical fact sheets of one smartphone. Try some iterative prompt engineering and do the following:

            1. Generate marketing product description for the smartphone

            2. Custom product description which has the following:

            ```
            The description should follow this format:

            Product Name: <Name of the smartphone>
            ​
            Description: <Brief Overview of the features>
            ​
            Product Specifications:
            <Table with key product feature specifications>
            ​
            The description should focus on the most important features
            a customer might look for in a phone including the foldable display screen, processing power, RAM, camera and battery life.
            ​
            After the description, the table should have the
            key specifications of the product. It should have two columns.
            The first column should have 'Feature'
            and the second column should have 'Specification'
            and try to put exact numeric values for features if they exist.
            Only put these features in the table - foldable display screen, processing power, RAM, camera and battery life
            ```

            3. Custom product description focusing on specific aspects like display, camera and in less than 60 words
        """

    @property
    def fact_sheet_contents(self) -> str:
        FileUtils.contents(file_name="fact_sheet_mobile", base_dir="exercises/exercise_3_social_media_marketing_analyst")

    def request_openai(self, prompt_txt: str) -> str:
        prompt = ChatPromptTemplate.from_template(prompt_txt)
        chain = prompt | self.openai
        response = chain.invoke({'fact_sheet_mobile': self.fact_sheet_contents})
        st.write(response.content)