import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from streamlit_chat import message

# Load environment variables from .env file
load_dotenv()

def get_secret(name):
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return os.getenv(name)


# API Keys and Model Configurations
HUGGINGFACE_API_KEY = get_secret("HUGGINGFACE_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")  # Groq API key for Part 1 and Part 3
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Use Hugging Face online model for Part 2

def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to .env or Streamlit secrets.")
    return Groq(api_key=GROQ_API_KEY)

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! How can I assist you?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]
# Part 0: Introduction
def display_introduction():
    st.title("Welcome to DocuBrain AI Assistant")
    st.write("""
    **DocuBrain** is a powerful tool that offers three primary functionalities:
    
    1. **Text Processing**: Summarize articles, extract highlights, generate Points of Minutes (PoM), and follow custom instructions for text processing.
    2. **InsightDoc AI Analyzer**: Upload PDFs for comparison, summarization, and search-based tasks.
    3. **Chat with Assistant**: Ask questions and chat with a powerful AI assistant for personalized responses.
    
    
    
    You can explore the functionalities by selecting an option from the sidebar.
    """)

    # Provide download link for the sample files
    sample_files = {
        "Part - 1 Sample Essay (essay.pdf)": "essay.pdf",
        "Part - 2 Document Search Sample (google.pdf)": "google.pdf",
        "Part - 2 Document Sample (tesla.pdf)": "tesla.pdf",
        "Part - 2 Document Sample (uber.pdf)": "uber.pdf",
        "Part - 2 Sample Questions (sample_question.pdf)": "sample_question.pdf"
    }

    # Provide download options for users
    st.write("### Download Sample Files to test:")
    for file_name, file_path in sample_files.items():
        st.download_button(
            label=f"Download {file_name}",
            data=open(file_path, "rb").read(),
            file_name=file_name,
            mime="application/pdf"
        )


# Part 1: Summary, Highlights, PoM, and Custom Instructions (Unchanged)
def process_text_with_groq(task, text):
    try:
        task_map = {
            "summary": "Summarize the following text.",
            "highlight": "Extract the highlights from the text.",
            "point of minutes": "Generate a concise point of minutes (PoM) for the following text.",
            "custom": "Follow the user's custom instructions for the given text."
        }
        user_prompt = f"{task_map[task]}: {text}"
        chat_completion = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error accessing Groq API: {e}"

def display_text_processing():
    st.write("### Text Processing ")
    st.write("Get summaries, highlights, PoM, or custom instructions for a given text or article.")
    task = st.selectbox(
        "Select Task",
        ["summary", "highlight", "point of minutes", "custom"],
        index=0
    )
    user_input = st.text_area(
        "Enter your text or article:",
        key='llama_input',
        height=200,
    )
    submit_button = st.button("Process")

    if submit_button and user_input:
        with st.spinner("Processing..."):
            output = process_text_with_groq(task, user_input)
            st.write("### Output:")
            st.success(output)

# Part 2: Document Search (Updated)
def create_conversational_chain(vector_store):
    llm = HuggingFaceHub(
        repo_id=MODEL_NAME,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        model_kwargs={"temperature": 0.75, "top_p": 0.9, "max_length": 1024}
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    full_answer = result["answer"].strip()

    # Crop the response to show only the relevant part (if applicable)
    if "Question:" in full_answer:
        question_start = full_answer.find("Question:")
        cropped_answer = full_answer[question_start:]
    else:
        cropped_answer = "The response does not contain a clearly defined question and answer."

    # Handle missing context explicitly
    if "I don't know" in cropped_answer or not cropped_answer.strip():
        cropped_answer = f"The provided context does not provide information on '{query}'."

    history.append((query, cropped_answer))
    return cropped_answer, full_answer, result.get("source_documents", [])

def display_document_search(chain):
    st.write("### Document Search")
    user_input = st.text_input("Ask a question about your document:", key='doc_input')
    submit_button = st.button("Search")

    if submit_button and user_input:
        with st.spinner("Searching..."):
            cropped_output, full_output, sources = conversation_chat(user_input, chain, st.session_state['history'])

            # Display cropped answer
            st.write("### Cropped Answer:")
            st.success(cropped_output)

            # Option to view full uncropped response
            with st.expander("View Full Response"):
                st.write(full_output)

            # Display related context
            if sources:
                with st.expander("Related Context"):
                    for doc in sources:
                        st.write(doc.page_content)

# Part 3: Dedicated Chat Window (Unchanged)
def chat_with_groq(text):
    try:
        chat_completion = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error accessing Groq API: {e}"

def display_chat_window():
    st.write("### Chat with Llama")
    reply_container = st.container()
    container = st.container()

    with container:
        user_input = st.text_input("Enter your message:", key='chat_input')
        submit_button = st.button("Send")

        if submit_button and user_input:
            with st.spinner("Generating response..."):
                output = chat_with_groq(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

# Main Functionality
def main():
    st.sidebar.title("App Navigation")
    app_mode = st.sidebar.radio(
        "Choose a mode",
        ["Part 0: Introduction and Guide","Part 1: Text Processing", "Part 2: InsightDoc AI Analyzer", "Part 3: Chat Window"]
    )

    initialize_session_state()
    if app_mode == "Part 0: Introduction and Guide":
        display_introduction()
    elif app_mode == "Part 1: Text Processing":
        display_text_processing()
    elif app_mode == "Part 2: InsightDoc AI Analyzer":
        st.subheader("InsightDoc AI Analyzer")
        st.write("""
            **DocuBrain**  Directly use the document (UPTO 500 PAGES) upload feature and perform tasks like comparison, summarization, and search-based queries.
            Use sample files from page 0.
            1. **Comparing Documents:** Upload one or more documents to compare specific details.  
            Example:  
            Compare how Tesla and Google incorporate AI into their respective business operations.

            2. **Searching Within Documents:** Upload documents to search for specific information.  
            Example:  
            Search for the revenue of Google in 2023?

            3. **Custom Tasks:** Upload documents and perform tailored tasks based on your needs.  
            Example:  
            Summarize the key points of a market research report.
        """)
        uploaded_files = st.sidebar.file_uploader("Upload PDF files:", accept_multiple_files=True)

        if uploaded_files:
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

            chain = create_conversational_chain(vector_store)
            display_document_search(chain)

    elif app_mode == "Part 3: Chat Window":
        display_chat_window()

if __name__ == "__main__":
    main()
