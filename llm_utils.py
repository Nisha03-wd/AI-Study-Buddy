import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document


@st.cache_resource
def get_embeddings_model():
    """Get the embeddings model from HuggingFace."""
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")


def build_qa_chain(text_data):
    """Builds a question-answering chain from text data."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text_data)

    documents = [Document(page_content=t) for t in texts]

    embeddings = get_embeddings_model()

    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        task="text2text-generation",
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )
    return qa_chain


def summarize_text(text_data):
    """Generates a summary for the given text."""
    llm = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",
        task="summarization",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={"max_length": 250, "min_length": 30},
    )

    docs = [Document(page_content=text_data)]

    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(docs)
    return summary


def generate_flashcards(text_data):
    """Generates flashcards from the text data."""
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        task="text2text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={"temperature": 0.7, "max_length": 1024},
    )

    prompt_template = """You are an expert in creating study materials.
    Given the following text from a lecture note, generate a set of flashcards.
    Each flashcard should have a 'Front' (a question or a term) and a 'Back' (the answer or definition).
    Format the output clearly with 'Front:' and 'Back:' for each card, separated by '---'.

    Text: "{text}"

    Flashcards:
    """

    prompt = PromptTemplate.from_template(prompt_template)
    truncated_text = text_data[:3000]
    chain = prompt | llm
    flashcards = chain.invoke({"text": truncated_text})
    return flashcards