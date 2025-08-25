import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the token
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Now you can use the hf_token variable in your code
# For example, when logging in or making API calls
# Note: Many huggingface_hub functions will automatically find this
# environment variable, so you might not even need to pass it explicitly.

# app.py
import streamlit as st
from backend.pdf_utils import extract_text_from_pdf
from backend.llm_utils import build_qa_chain, summarize_text, generate_flashcards


# Cache the QA chain creation for efficiency
@st.cache_resource
def get_qa_chain(text):
    return build_qa_chain(text)

st.set_page_config(page_title="AI Study Buddy", layout="wide")

st.title("📚 AI Study Buddy")
st.write("Upload lecture notes (PDF/text), then ask questions, get summaries, or flashcards!")

uploaded_file = st.file_uploader("Upload your lecture notes (PDF)", type=["pdf"])
text_data = ""

if uploaded_file is not None:
    text_data = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded & processed!")
else:
    text_data = st.text_area("Or paste your notes here:")

if text_data:
    if st.button("🔁 Rebuild knowledge base"):
        # Clear the cache for the chain-building function
        get_qa_chain.clear()

    qa_chain = get_qa_chain(text_data)

    st.subheader("Ask Questions")
    user_question = st.text_input("Type your question:")
    if st.button("Get Answer") and user_question:
        response = qa_chain({"question": user_question})
        st.write("**Answer:** ", response["answer"])

    st.subheader("Generate Summary")
    if st.button("Summarize Notes"):
        summary = summarize_text(text_data)
        st.write("### ✨ Summary")
        st.write(summary)

    st.subheader("Generate Flashcards")
    if st.button("Make Flashcards"):
        flashcards = generate_flashcards(text_data)
        st.write("### 🃏 Flashcards")
        st.write(flashcards)
