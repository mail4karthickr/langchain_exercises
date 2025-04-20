import streamlit as st
import pandas as pd
from langchain_core.prompts import PromptTemplate
from llm.openai import LLM
from .it_support_response import ITSupportResponse
from langchain_core.output_parsers import JsonOutputParser

def run():
    ITSupportAnalyst().run()

class ITSupportAnalyst:
    def __init__(self):
        self.llm = LLM().chat_groq

    def run(self):
        st.markdown(self.project_description)
        user_input = st.text_area(
            label="Enter support tickets (comma-separated)",
            value=self.default_text,
            height=200
        )
        if st.button("Analyze Tickets"):
            tickets = [ticket.strip() for ticket in user_input.split(",") if ticket.strip()]
            # st.write(tickets)
            self.ticket_analyst(tickets) 

    def ticket_analyst(self, it_support_queue: list[str]) -> list[str: str]:
        prompt_txt = """
            Act as an Information Technology (IT) customer support agent.
            For the IT support message mentioned below
            Use the following output format when generating the output response

            Output format instructions:
            {format_instructions}

            Customer IT support message:
            {it_support_msg}
        """
        formatted_msgs = [{'it_support_msg': msg} for msg in it_support_queue]
        parser = JsonOutputParser(pydantic_object=ITSupportResponse)
        prompt = PromptTemplate(
            template=prompt_txt,
            input_variables=['it_support_msg'],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )
        chain = prompt | self.llm | parser
        # response type list[Dict]
        responses = chain.map().invoke(formatted_msgs)
        df = pd.DataFrame(responses)
        st.dataframe(df, width=1000)
        # print(f"Output type {type(response)}")
    
    @property
    def project_description(self):
        return """
        ## Mini-Project 4 - IT Support Analyst

        Ask ChatGPT to act as a IT support agent, process each customer IT ticket message and output the response in JSON with the following fields

        ```
        orig_msg: The original customer message
        orig_lang: Detected language of the customer message e.g. Spanish
        category: 1-2 word describing the category of the problem
        trans_msg: Translated customer message in English
        response: Response to the customer in orig_lang
        trans_response: Response to the customer in English
        ```

        Try to use a JSON parser to get the responses in JSON for each ticket
        """
    @property
    def default_text(self):
        return (
            "Não consigo sincronizar meus contatos com o telefone. Sempre recebo uma mensagem de falha., "
            "Ho problemi a stampare i documenti da remoto. Il lavoro non viene inviato alla stampante di rete., "
            "プリンターのトナーを交換しましたが、印刷品質が低下しています。サポートが必要です。, "
            "Я не могу войти в систему учета времени, появляется сообщение об ошибке. Мне нужна помощь., "
            "Internet bağlantım çok yavaş ve bazen tamamen kesiliyor. Yardım eder misiniz?, "
            "Не могу установить обновление безопасности. Появляется код ошибки. Помогите, пожалуйста."
        )