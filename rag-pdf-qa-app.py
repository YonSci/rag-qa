# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import LangSmith for tracing
from langsmith import traceable

# Import the RetrievalQA chain builder for RAG pipelines
from langchain.chains import RetrievalQA  

# Import a callback that streams LLM tokens to stdout in real time
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  

# Import the manager to coordinate multiple callback handlers
from langchain.callbacks.manager import CallbackManager  

# Import the new Ollama LLM wrapper from the standalone langchain-ollama package
from langchain_ollama import OllamaLLM

# Import the new Ollama embeddings class for generating embeddings via Ollama
from langchain_ollama import OllamaEmbeddings

# Import Chroma vector store wrapper (from langchain-chroma integration)
from langchain_chroma import Chroma

# Import the recursive text splitter that chunks long documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  

# Import document loaders for various file types and web content
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, WebBaseLoader

# Import PromptTemplate for building parameterized prompts
from langchain.prompts import PromptTemplate  

# Import the in-memory conversation buffer to track chat history
from langchain.memory import ConversationBufferMemory  

# Import Streamlit for building the web UI
import streamlit as st  

# Import time for simulating typing delays
import time  

# Import warnings to suppress deprecation notices
import warnings  

# Import hashlib for computing file hashes
import hashlib

# Import chromadb for direct access to Chroma client functionality
import chromadb

# Import validators for URL validation
import validators

# Suppress LangChain deprecation warnings that use UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=". *LangChainDeprecationWarning.*"
)

# Set Streamlit page configuration for wide layout
st.set_page_config(
    page_title="Document & Web Chatbot",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS for light theme UI
st.markdown("""
    <style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f7fa;
    }
    h1 {
        color: #2c3e50;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .chatbot-title {
        color: #3498db;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h3 {
        color: #34495e;
        font-size: 1.5em;
        margin-top: 20px;
    }
    .content-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #e6f3ff;
    }
    .stChatMessage.assistant {
        background-color: #f0f0f0;
    }
    .css-1lcbmhc {
        background-color: #34495e;
        color: white;
    }
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3 {
        color: white;
    }
    .css-1lcbmhc .stButton>button {
        background-color: #e74c3c;
    }
    .css-1lcbmhc .stButton>button:hover {
        background-color: #c0392b;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        border: 1px solid #bdc3c7;
        padding: 10px;
    }
    .stAlert {
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Ensure directories exist
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('data'):
    os.mkdir('data')

# Function to compute content hash
def compute_hash(content):
    if isinstance(content, str):
        content = content.encode('utf-8')
    sha256 = hashlib.sha256()
    sha256.update(content)
    return sha256.hexdigest()

# LangSmith RAG pipeline function with tracing
@traceable
def run_rag_pipeline(inputs):
    question = inputs["question"]
    docs = st.session_state.retriever.invoke(question)
    response = st.session_state.qa_chain.invoke({"query": question})
    return {"answer": response["result"]}

# Initialize prompt template
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

# Initialize conversation memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

# Initialize Chroma client and vectorstore
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path='data')

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory='data',
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    st.subheader("Select Language Model")
    model_dict = {
        "Tiny-llama": "tinyllama:latest",
        "Deepseek-r1": "deepseek-r1:1.5b",
        "Llama3": "llama3.2:3b",
        "Qwen3": "qwen3:4b",
        "Gemma": "gemma3:4b",
        "Mistral": "mistral:7b"
    }
    if 'selected_model_name' not in st.session_state:
        st.session_state.selected_model_name = "Llama3"
    selected_model_name = st.selectbox(
        "Choose a model",
        list(model_dict.keys()),
        index=list(model_dict.keys()).index(st.session_state.selected_model_name)
    )
    st.session_state.selected_model_name = selected_model_name
    selected_model_id = model_dict[selected_model_name]
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.success("Chat history cleared!")

# Initialize or update LLM
if 'llm' not in st.session_state or st.session_state.current_model != selected_model_id:
    try:
        st.session_state.llm = OllamaLLM(
            base_url="http://localhost:11434",
            model=selected_model_id,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        st.session_state.current_model = selected_model_id
        if 'qa_chain' in st.session_state:
            del st.session_state.qa_chain
        st.success(f"✅ Switched to {selected_model_name} model.")
    except Exception as e:
        st.error(f"Failed to load model {selected_model_name}: {str(e)}")

# Main content
with st.container():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="chatbot-title">SmartDoc Chatbot</div>', unsafe_allow_html=True)
    
    st.markdown("### 🚀 How to Use the Chatbot")
    st.markdown("""
    1. **Select a Language Model**: Choose a model from the sidebar (e.g., Llama3).
    2. **Upload a Document or Enter a URL**: Upload a file or scrape a URL.
    3. **Wait for Processing**: A progress bar will show processing status.
    4. **Ask Questions**: Type questions in the chat input.
    5. **Manage Settings**: Switch models or clear chat history in the sidebar.
    """)
    
    st.markdown("### 📤 Upload a Document or Enter a URL")
    st.info("📎 Supported file types: PDF, DOCX, TXT, Markdown, HTML | Enter a URL to scrape web content")

    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'md', 'html'])

    url = st.text_input("Or enter a URL to scrape:", placeholder="https://example.com")
    if url and st.button("Scrape URL"):
        if not validators.url(url):
            st.error("Invalid URL format.")
        else:
            with st.spinner("🔄 Scraping web content..."):
                try:
                    loader = WebBaseLoader(url)
                    data = loader.load()
                    web_content = "".join(doc.page_content for doc in data)
                    content_hash = compute_hash(web_content)
                    collections = st.session_state.chroma_client.list_collections()
                    collection_names = [col.name for col in collections]
                    if content_hash in collection_names:
                        st.session_state.vectorstore = Chroma(
                            persist_directory='data',
                            collection_name=content_hash,
                            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                        )
                        st.success("✅ Loaded existing web content data.")
                    else:
                        progress_bar = st.progress(0)
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1500,
                            chunk_overlap=200
                        )
                        all_splits = text_splitter.split_documents(data)
                        progress_bar.progress(60)
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=all_splits,
                            embedding=OllamaEmbeddings(model="nomic-embed-text"),
                            collection_name=content_hash,
                            persist_directory='data'
                        )
                        progress_bar.progress(100)
                        st.success("✅ Web content processed and stored.")
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
                    if 'qa_chain' in st.session_state:
                        del st.session_state.qa_chain
                except Exception as e:
                    st.error(f"Failed to scrape or process URL: {str(e)}")

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        file_hash = compute_hash(bytes_data)
        file_path = f"files/{uploaded_file.name}"
        if not os.path.isfile(file_path):
            with open(file_path, "wb") as f:
                f.write(bytes_data)
        collections = st.session_state.chroma_client.list_collections()
        collection_names = [col.name for col in collections]
        if file_hash in collection_names:
            st.session_state.vectorstore = Chroma(
                persist_directory='data',
                collection_name=file_hash,
                embedding_function=OllamaEmbeddings(model="nomic-embed-text")
            )
            st.success(f"✅ Loaded existing {uploaded_file.name.split('.')[-1].upper()} data.")
        else:
            with st.spinner(f"🔄 Processing {uploaded_file.name.split('.')[-1].upper()}..."):
                progress_bar = st.progress(0)
                file_extension = uploaded_file.name.split('.')[-1].lower()
                try:
                    if file_extension == 'pdf':
                        loader = PyPDFLoader(file_path)
                    elif file_extension == 'docx':
                        loader = Docx2txtLoader(file_path)
                    elif file_extension == 'txt':
                        loader = TextLoader(file_path)
                    elif file_extension == 'md':
                        loader = UnstructuredMarkdownLoader(file_path)
                    elif file_extension == 'html':
                        loader = UnstructuredHTMLLoader(file_path)
                    else:
                        st.error(f"Unsupported file type: {file_extension}")
                        st.stop()
                    data = loader.load()
                    progress_bar.progress(30)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200
                    )
                    all_splits = text_splitter.split_documents(data)
                    progress_bar.progress(60)
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=all_splits,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        collection_name=file_hash,
                        persist_directory='data'
                    )
                    progress_bar.progress(100)
                    st.success(f"✅ {uploaded_file.name.split('.')[-1].upper()} processed and stored.")
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                    st.stop()
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'retriever' in st.session_state and st.session_state.retriever is not None:
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory
                }
            )

        st.markdown("### 💬 Ask a Question")
        query = st.chat_input("Ask a question about your document or web content:", key="user_input")
        
        if query:
            user_message = {"role": "user", "message": query}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(f"**You:** {query}")
            with st.chat_message("assistant"):
                with st.spinner(f"🤖 {selected_model_name} is thinking..."):
                    response = run_rag_pipeline({"question": query})
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response["answer"].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                if "```" in full_response:
                    code_blocks = full_response.split("```")
                    for i, block in enumerate(code_blocks):
                        if i % 2 == 1:
                            st.code(block.strip(), language="python")
            chatbot_message = {"role": "assistant", "message": response["answer"]}
            st.session_state.chat_history.append(chatbot_message)

    else:
        if not url:
            st.write("Please upload a document or enter a URL to start chatting.")

    st.markdown("### 📜 Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])
    
    st.markdown('</div>', unsafe_allow_html=True)