import os
import streamlit as st
import importlib

def setup_api_key() -> bool:
    # Try loading from .env first
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = env_key
        return True
    
    # Ask user to input manually
    if "openai_api_key" not in st.session_state:
        if "openai_api_key" not in st.session_state:
            st.title("🔐 Enter OpenAI API Key to Start")
            openai_api_key_input = st.text_input("Enter your OpenAI API Key", type="password")

            if st.button("Submit"):
                if openai_api_key_input.strip():
                    st.session_state.openai_api_key = openai_api_key_input.strip()
                    st.success("API Key stored! You can now use the app.")
                    st.rerun()
                else:
                    st.warning("Please enter a valid key")

    return "openai_api_key" in st.session_state

# 🛑 Require API Key before anything else
if not setup_api_key():
    st.stop()

# Sidebar Navigation
st.sidebar.title("📚 LangChain Exercises")

excersise_options = {
    "Customer Review Analyst": "exercises.exercise_1_review_analyst.main",
    "Research Paper Analyst": "exercises.exercise_2_research_paper_analyst.main",
    "Social Media Marketing Analyst": "exercises.exercise_3_social_media_marketing_analyst.main"
}

selected_excercise = st.sidebar.radio("Select an excersise:", list(excersise_options.keys()))


if selected_excercise:
    module_path = excersise_options[selected_excercise]
    module = importlib.import_module(module_path)
    module.run()