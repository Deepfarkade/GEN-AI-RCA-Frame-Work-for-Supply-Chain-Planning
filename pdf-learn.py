import streamlit as st
import os
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain import HuggingFaceHub
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ChatMessageHistory
import streamlit as st 
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS

TF_ENABLE_ONEDNN_OPTS=0


def create_llms_model():
    llm = CTransformers(model="mistral-7b-instruct-v0.1.Q2_K.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm




@st.cache_resource
def qa_llm():
    llm = create_llms_model()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,memory=memory )
    return qa


def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    
    
    return answer, generated_text







# Initialize Streamlit app
st.title("Gen AI Chatbot")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)


# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Please Ask me your Query 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Create memory
memory = ConversationBufferWindowMemory(memory_key="chat_history",k=2, return_messages=True)


# Define chat function
def conversation_chat(query):
    result = process_answer(query)
    
    # Extract relevant information from the result
    answer, generated_text = result
    
    # Print the generated_text
    print("Generated Text:", generated_text)
    
    # Save chat history to session state
    st.session_state['history'].extend(generated_text["chat_history"])
    
    return answer

# Display chat history
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Ask your Query here", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
            
   