from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import textract
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
from PIL import Image

# Initialize Streamlit page
img = Image.open(r"title_image.png")
st.set_page_config(page_title="FileWise: Empowering Insights, Effortlessly.", page_icon=img)
st.header("Ask Your File📄")
file = st.file_uploader("Upload your file")

# Initialize session state
if 'context' not in st.session_state:
    st.session_state.context = {
        'knowledge_base': None,
        'qa_chain': None,
        'conversation_history': [],
    }

if file is not None:
    content = file.read()  # Read the file content once

    if file.type == 'application/pdf':
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == 'text/plain':
        text = content.decode('utf-8')
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = textract.process(content)

    if st.session_state.context['knowledge_base'] is None:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        st.session_state.context['knowledge_base'] = knowledge_base

    query = st.text_input("Ask your Question about the file")
    if query:
        docs = st.session_state.context['knowledge_base'].similarity_search(query)

        if st.session_state.context['qa_chain'] is None:
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            st.session_state.context['qa_chain'] = chain

        response = st.session_state.context['qa_chain'].run(input_documents=docs, question=query)
        
        # Store conversation history
        st.session_state.context['conversation_history'].append({
            'query': query,
            'response': response,
        })

        st.success(response)

# Provide options to start a new conversation or clear context
if st.button("Start a New Conversation"):
    st.session_state.context['knowledge_base'] = None
    st.session_state.context['qa_chain'] = None
    st.session_state.context['conversation_history'] = []

if st.button("Clear Context"):
    st.session_state.context['knowledge_base'] = None
    st.session_state.context['qa_chain'] = None
    st.session_state.context['conversation_history'] = []
