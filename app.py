import os
import streamlit as st
import importlib

def setup_api_key() -> bool:
    if "openai_api_key" in st.session_state and "groq_api_key" in st.session_state:
        # Keys are already entered → no need to show input
        return True

    # Show input fields
    with st.form("api_key_form"):
        st.title("🔐 Enter API Keys to Start")

        openai_api_key_input = st.text_input("Enter your OpenAI API Key", type="password")
        groq_api_key_input = st.text_input("Enter your GROQ API Key", type="password")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if openai_api_key_input.strip() and groq_api_key_input.strip():
                st.session_state.openai_api_key = openai_api_key_input.strip()
                os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                st.session_state.groq_api_key = groq_api_key_input.strip()
                st.success("API Keys saved! Please rerun or continue.")
                st.experimental_rerun()
            else:
                st.warning("Please enter both API keys.")

    # Since keys are not set yet, return False
    return False

# 🛑 Require API Key before anything else
if not setup_api_key():
    st.stop()

st.sidebar.title("🤖 Agentic AI Exercises")

# Exercise dictionaries
langchain_exercises = {
    # langchain
    "Customer Review Analyst": "exercises.exercise_1_review_analyst.main",
    "Research Paper Analyst": "exercises.exercise_2_research_paper_analyst.main",
    "Social Media Marketing Analyst": "exercises.exercise_3_social_media_marketing_analyst.main",
    "IT Support Analyst": "exercises.exercise_4_it_support_analyst.main",
    "ChatBot": "exercises.exercise_5_conversation_memory.main",
    "Linking Multiple Chains": "exercises.exercise_6_linking_multiple_chains.main",
    "Branching and Merging Chains": "exercises.exercise_7_branching_merging_chains.main",
    "Routing Chains": "exercises.exercise_8_routing_chains.main",
    "Product Recommender": "exercises.exercise_9_product_recommender.main",
    # prompt_engineering
    "Persona Pattern": "exercises.exercise_10_persona_pattern.main",
    "Flipped Interaction Pattern": "exercises.exercise_11_flipped_interaction_pattern.main",
    "N-Shot Prompting Pattern": "exercises.exercise_12_n_shot_prompting_pattern.main",
    "Directional Stimulus Pattern": "exercises.exercise_13_directional_stimulus_pattern.main",
    "Template Pattern": "exercises.exercise_14_template_pattern.main",
    "Meta Language Pattern": "exercises.exercise_15_meta_lang_pattern.main",
    "Chain-of-Thought Pattern": "exercises.exercise_16_chain_of_thought_pattern.main",
    "Self-Consistency Pattern": "exercises.exercise_17_self_consistency_pattern.main",
    "Least-to-Most Pattern": "exercises.exercise_18_least_to_most_pattern.main",
    "ReAct Pattern": "exercises.exercise_19_react_pattern.main",
    "Financial Statement Summarizer": "exercises.exercise_20_summarize_financial_statement.main",
    #rag
    "Intro": "exercises.3_rag_systems_essentials.main",
    "Retrievers": "exercises.3_rag_systems_essentials.retrievers.main",
    "Similarity or Ranking based Retrieval ": "exercises.exercise_21_similarity_ranking_based.main",
    "Multi Query Retrieval": "exercises.exercise_22_multi_query_retrieval.main",
    "Content Compression Retriever": "exercises.exercise_23_contextual_compression_retriever.main",
    "Wikipedia Search Engine": "exercises.exercise_24_search_engine.main"
}

# Section 1: Langchain
st.sidebar.markdown("#### 🔗 Langchain")
selected_langchain = st.sidebar.radio(
    "Select LangChain Exercise",
    list(langchain_exercises.keys()),
    key="selected_exercise"
)

# Separator between sections
# st.sidebar.markdown("---")

# Prompt Engineering group
# st.sidebar.markdown("#### 🧠 Prompt Engineering")
# selected_prompt = st.sidebar.radio(
#     "Select Prompt Engineering Exercise",
#     list(prompt_engineering_exercises.keys()),
#     key="prompt_engineering_exercise",
#     on_change=on_select_prompt
# )

# # RAG group
# st.sidebar.markdown("#### 📚 RAG (Retrieval-Augmented Generation)")
# selected_prompt = st.sidebar.radio(
#     "RAG",
#     list(rag_systems_essentials.keys()),
#     key="selected_rag_systems_essentials",
#     on_change=on_select_rag_system_essentials
# )

# Logic to decide which module to load
# selected_exercise_path = None

if st.session_state.selected_exercise:
    selected_exercise_path = langchain_exercises[st.session_state.selected_exercise]
    # Load and run
    if selected_exercise_path:
        if selected_exercise_path.strip():
            module = importlib.import_module(selected_exercise_path)
            module.run()
        else:
            st.warning("⚠️ Exercise not implemented yet.")