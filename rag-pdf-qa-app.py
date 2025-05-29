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

# Import the PDF loader from the community package to extract pages as Documents
from langchain_community.document_loaders import PyPDFLoader

# Import PromptTemplate for building parameterized prompts
from langchain.prompts import PromptTemplate  

# Import the in-memory conversation buffer to track chat history
from langchain.memory import ConversationBufferMemory  

# Import Streamlit for building the web UI
import streamlit as st  

# Import os for file‐ and directory‐level operations
import os  

# Import time for simulating typing delays
import time  

# Import warnings to suppress deprecation notices
import warnings  

# Suppress LangChain deprecation warnings that use UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,               # Target LangChain’s deprecation warnings
    message=". *LangChainDeprecationWarning.*"
)

# Ensure there’s a 'files' folder to store uploaded PDFs
if not os.path.exists('files'):
    os.mkdir('files')

# Ensure there’s a 'data' folder to store vector‐store persistence
if not os.path.exists('data'):
    os.mkdir('data')

# On first run, store our prompt template string in session_state
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

# Initialize a PromptTemplate from that template if not already done
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],  # placeholders we’ll fill
        template=st.session_state.template,                  # the string defined above
    )

# Initialize our ConversationBufferMemory to keep track of past messages
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",      # where to store the chat history in the chain
        return_messages=True,      # always return the full message list
        input_key="question"       # the input variable name for the user’s question
    )

# Load (or initialize) a Chroma vectorstore pointing at the 'data' folder
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory='data',                             # where Chroma stores its files
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")  # how to embed text
    )

# Instantiate the local Ollama LLM client if not already in session_state
if 'llm' not in st.session_state:
    st.session_state.llm = OllamaLLM(
        base_url="http://localhost:11434",                   # Ollama server endpoint
        model="llama3.2:3b",                                 # which Ollama model to use
        verbose=True,                                        # print detailed logs
        callback_manager=CallbackManager(                    # manage streaming callbacks
            [StreamingStdOutCallbackHandler()]
        ),
    )

# Initialize an empty chat history list in session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set the title of the Streamlit app
st.title("PDF Chatbot")

# Render a file uploader widget that accepts only PDF files
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

# Re-display all past messages in the Streamlit chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# If the user has uploaded a PDF, process it
if uploaded_file is not None:

    # Only process if we haven’t already saved this PDF locally
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):

            # Read the uploaded bytes and write them to disk
            bytes_data = uploaded_file.read()
            with open("files/"+uploaded_file.name+".pdf", "wb") as f:
                f.write(bytes_data)

            # Load the PDF pages as LangChain Documents
            loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            # Create a text splitter to chunk the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,     # max chars per chunk
                chunk_overlap=200,   # overlap to retain context
                length_function=len  # function to measure chunk length
            )
            all_splits = text_splitter.split_documents(data)

            # Build a new vector store from those chunks and persist to disk
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="nomic-embed-text")
            )
            st.session_state.vectorstore.persist()

    # Wrap the vectorstore in a retriever for similarity search
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    # Initialize the RetrievalQA chain if it’s not yet created
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,           # the LLM to generate answers
            chain_type='stuff',                # simple “stuff-in-prompt” approach
            retriever=st.session_state.retriever,  
            verbose=True,  
            chain_type_kwargs={
                "verbose": True,                  # more internal logging
                "prompt": st.session_state.prompt,# our PromptTemplate
                "memory": st.session_state.memory,# conversation memory
            }
        )

    # Display a chat input box and capture the user’s query
    if query := st.chat_input("You:", key="user_input"):
        # Append the user message to our chat history
        user_message = {"role": "user", "message": query}
        st.session_state.chat_history.append(user_message)

        # Render the user’s message in the UI
        with st.chat_message("user"):
            st.markdown(query)

        # As the assistant, show a spinner while we're generating
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                # Invoke the RAG pipeline to get a response
                response = st.session_state.qa_chain.invoke({"query": query})

            # Stream the response token-by-token with a blinking cursor
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)  # small delay for typing effect
                message_placeholder.markdown(full_response + "▌")
            # Final render without the cursor
            message_placeholder.markdown(full_response)

        # Save the assistant’s reply in chat history
        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)

# If no PDF has been uploaded yet, prompt the user to do so
else:
    st.write("Please upload a PDF file.")
