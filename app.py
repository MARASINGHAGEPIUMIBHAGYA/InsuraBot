import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine  # Your custom RAG implementation

# Configure page
st.set_page_config(
    page_title="🏡 Home Insurance Assistant",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for chat UI
st.markdown("""
    <style>
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            max-width: 80%;
            word-wrap: break-word;
        }
        .chat-message.user {
            background-color: #e6f2ff;
            margin-left: auto;
            text-align: right;
            border: 1px solid #cce0ff;
        }
        .chat-message.assistant {
            background-color: #f9f9f9;
            margin-right: auto;
            border: 1px solid #e0e0e0;
        }
        .chat-message p {
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
        }
        .chat-message strong {
            font-size: 0.9rem;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I'm your Home Insurance Assistant. Ask me anything about your policy coverage, claims process, or policy details."
    }]

# Load and initialize RAG engine
if "rag" not in st.session_state:
    with st.spinner("Loading policy documents..."):
        try:
            load_dotenv()
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                st.error("API key not found. Please check your .env file")
                st.stop()
                
            genai.configure(api_key=GEMINI_API_KEY)
            st.session_state.rag = RAGEngine("data/Home_insurance_sample.pdf")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()

# Sidebar
with st.sidebar:
    st.title("🏡 Home Insurance")
    st.markdown("""
    ### About this assistant
    I can help you understand:
    - Policy coverage details
    - Claims process
    - Premium information
    - Exclusions and limitations
    
    Ask me anything about your home insurance policy!
    """)
    st.markdown("---")
    st.markdown("### Sample Questions")
    sample_questions = [
        "What does my policy cover for water damage?",
        "How do I file a claim for roof damage?",
        "What is my deductible for fire damage?",
        "Are home office equipment covered?",
        "What's the process for emergency repairs?"
    ]
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}"):
            st.session_state.user_input = question

# Main Chat Title
st.title("🏡 Home Insurance Assistant")

# Display Chat Messages
for message in st.session_state.messages:
    with st.container():
        role = message["role"]
        content = message["content"]
        css_class = "user" if role == "user" else "assistant"
        sender = "You" if role == "user" else "Assistant"
        st.markdown(f"""
            <div class="chat-message {css_class}">
                <strong>{sender}:</strong>
                <p>{content}</p>
            </div>
        """, unsafe_allow_html=True)

# Chat Input Form
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input(
    label="",  
    value=st.session_state.get("user_input", ""),
    key="user_input",
    placeholder="Type your question here...",
    label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Searching policy documents..."):
            try:
                # API connection check
                try:
                    genai.list_models()  # basic connectivity test
                except Exception as api_error:
                    st.error("API connection failed. Please check your API key.")
                    raise api_error

                # Query RAG engine
                response = st.session_state.rag.ask(user_input)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}. Please try again later."
                })
                st.rerun()
