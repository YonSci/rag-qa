import os
import time
import warnings

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import (
    TokenTextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings

# ─────────────────────────────────────────────────────────────────────────────
# Suppress LangChain deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*LangChainDeprecationWarning.*"
)

# ─────────────────────────────────────────────────────────────────────────────
# Ensure folders exist
for folder in ("files", "data"):
    os.makedirs(folder, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialization
if "template" not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

Context: {context}
History: {history}

User: {question}
Chatbot:"""

if "prompt" not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory="data",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )

if "llm" not in st.session_state:
    st.session_state.llm = OllamaLLM(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────────────────────────────────────
# App UI
st.title("PDF Chatbot")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])

# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    local_path = f"files/{uploaded_file.name}.pdf"

    # Save PDF locally if new
    if not os.path.isfile(local_path):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            with open(local_path, "wb") as f:
                f.write(bytes_data)

    # Always load the PDF into 'data'
    loader = PyPDFLoader(local_path)
    data = loader.load()

    # ─────────────────────────────────────────────────────────────────────────
    # Chunking options in the main body
    st.markdown("### Chunking Options")

    strategy = st.selectbox(
        "Splitter strategy",
        [
            "CharacterTextSplitter",
            "RecursiveCharacterTextSplitter",
            "TokenTextSplitter",
            "SentenceTransformersTokenTextSplitter",
            "SpacyTextSplitter",
            "SemanticChunker",
        ],
    )
    chunk_size = st.number_input(
        "Chunk size",
        min_value=50,
        max_value=5000,
        value=1500,
        step=50,
        help="Max tokens/characters per chunk",
    )
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=chunk_size - 1,
        value=200,
        step=10,
        help="How much adjacent chunks overlap",
    )

    # Instantiate the chosen splitter (but don’t run it yet)
    if strategy == "CharacterTextSplitter":
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "RecursiveCharacterTextSplitter":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "TokenTextSplitter":
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "SentenceTransformersTokenTextSplitter":
        splitter = SentenceTransformersTokenTextSplitter(
            model_name="all-MiniLM-L6-v2",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "SpacyTextSplitter":
        splitter = SpacyTextSplitter(
            pipeline="en_core_web_sm",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:  # SemanticChunker
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Button to execute splitting
    if st.button("Run Chunking"):
        with st.spinner("Splitting document..."):
            all_splits = splitter.split_documents(data)

        # Show number of chunks generated
        st.success(f"Generated **{len(all_splits)}** chunks using **{strategy}**")

        # Rebuild vector store on those chunks
        st.session_state.vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
        )

        # Initialize retriever & QA chain if needed (or update retriever)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff",
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                },
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Chat input & response (only active after chunking)
    if "qa_chain" in st.session_state and (query := st.chat_input("You:", key="user_input")):
        st.session_state.chat_history.append({"role": "user", "message": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"), st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain.invoke({"query": query})
            placeholder = st.empty()
            full_text = ""
            for token in response["result"].split():
                full_text += token + " "
                time.sleep(0.05)
                placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)

        st.session_state.chat_history.append(
            {"role": "assistant", "message": response["result"]}
        )

else:
    st.write("Please upload a PDF file.")
