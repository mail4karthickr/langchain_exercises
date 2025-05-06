import streamlit as st
from pydantic import BaseModel, Field
from functools import cached_property

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers import ConsoleCallbackHandler

from llm.openai import LLM  # This should return openai_gpt4o as per your setup


class ProjectPlanResponse(BaseModel):
    question: str = Field(..., description="The question to ask the user")
    projectPlan: str = Field(..., description="Final plan or empty if still asking")
    isQuestion: bool = Field(..., description="True if asking a question, False if giving final plan")


class FlippedInteractionPattern:
    def __init__(self):
        self.model = LLM().openai_gpt4o  # Your GPT-4o model wrapper

    def run(self):
        st.markdown(self.flipped_interactioin_definition())

        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = ChatMessageHistory()
        if "conversation" not in st.session_state:
            st.session_state.conversation = []  # (speaker, message)
        if "next_prompt" not in st.session_state:
            st.session_state.next_prompt = None
        if "conversation_ended" not in st.session_state:
            st.session_state.conversation_ended = False
        if "final_plan" not in st.session_state:
            st.session_state.final_plan = ""

        conv_chain = self.chat_chain(st.session_state.chat_history)
        st.session_state.conv_chain = conv_chain

        # Start conversation
        if st.button("🛠️ Start Project Plan") and not st.session_state.conversation:
            ai_response = conv_chain.invoke(
                {"input": self.prompt_txt},
                config={"configurable": {"session_id": "planning"}}
            )
            st.session_state.conversation.append(("🤖 AI", ai_response.question))
            st.session_state.next_prompt = ai_response

            if not ai_response.isQuestion:
                st.session_state.conversation_ended = True
                st.session_state.final_plan = ai_response.projectPlan

        # User input form (loop conditionally continues)
        if not st.session_state.conversation_ended and st.session_state.next_prompt:
            with st.form("user_input_form"):
                user_input = st.text_area("💬 Your Answer")
                submitted = st.form_submit_button("Submit")

                if submitted and user_input.strip():
                    # Save user message
                    st.session_state.conversation.append(("🙋 You", user_input))

                    # Get next AI response
                    ai_response = conv_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "planning"}}
                    )

                    # Store response
                    if ai_response.isQuestion:
                        st.session_state.conversation.append(("🤖 AI", ai_response.question))
                        st.session_state.next_prompt = ai_response
                    else:
                        st.session_state.conversation.append(("🤖 AI", "✅ Final project plan ready"))
                        st.session_state.final_plan = ai_response.projectPlan
                        st.session_state.next_prompt = None
                        st.session_state.conversation_ended = True

        # Display full conversation
        st.divider()
        st.subheader("🧠 Project Planning Conversation")

        for speaker, text in st.session_state.conversation:
            st.markdown(f"**{speaker}:** {text}")

        if st.session_state.conversation_ended:
            st.success("✅ Planning complete! Final project plan:")
            st.text(st.session_state.final_plan)

    def chat_chain(self, history):
        parser = PydanticOutputParser(pydantic_object=ProjectPlanResponse)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.full_sys_prompt_with_format_instructions),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.model | parser

        return RunnableWithMessageHistory(
            chain,
            lambda session_id: history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    @cached_property
    def prompt_txt(self):
        return """
        You are an expert at generating project plans for data science projects. 
        You will gather inputs from the user step-by-step by asking one question at a time. After you gather enough details, generate a final project plan.

        ⚠️ You MUST respond in exactly this JSON format:

        {
        "isQuestion": true | false,
        "question": "string (ask this if isQuestion is true, else leave empty)",
        "projectPlan": "string (final plan text if isQuestion is false, else leave empty)"
        }

        ⚠️ Strict Rules:
        - ✅ Only ask ONE question at a time.
        - ✅ Return ONLY a single JSON object.
        - ❌ Do NOT include markdown, bullet points, headers, explanations, or multiple questions.
        - ❌ Do NOT repeat steps or output summaries.
        - ✅ If you are ready to give the final plan, set `isQuestion` to false and give the plan in `projectPlan`.

        Begin by asking your first question.
        """

    @cached_property
    def full_sys_prompt_with_format_instructions(self):
        parser = PydanticOutputParser(pydantic_object=ProjectPlanResponse)
        format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

        return f"""
    You are an expert assistant for project planning.

    You must follow this strict JSON response format for EVERY reply:

    {format_instructions}

    Rules:
    - Ask only one question at a time.
    - Do not explain anything.
    - Do not include markdown or code blocks.
    - Always return a valid JSON object.
    """

    def flipped_interactioin_definition(self):
        return """
## 🔄 Flipped Interaction Pattern

In this pattern, the AI initiates the conversation by asking questions to gather information.

Once it has enough input, it will create a project plan and end the conversation.

You're participating in a smart assistant–led planning session!
"""


# Entry point
def run():
    FlippedInteractionPattern().run()
