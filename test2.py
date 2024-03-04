import streamlit as st
import os
import pandas as pd
import time
from streamlit_chat import message
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from utils.setup import setup_connexion, setup_session_state
from utils.vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
)

os.environ['REQUESTS_CA_BUNDLE'] = 'cert.crt'

setup_connexion()


TF_ENABLE_ONEDNN_OPTS=0



def process_and_display_sql(question):
    sql = generate_sql_cached(question=question)
    if sql:
        # Process SQL query 
        if sql!="":
            df = run_sql_cached(sql=sql)
            output = "SQL Engine Result: " + str(df)

            # Save to memory
            memory.save_context({"input": question}, {"output": output})

            return df
    else:
        # If no SQL query is generated, you may want to handle this case accordingly
        return "Sorry, I couldn't generate a valid SQL query for your question."
    
# follow up questions function

def followup_questions(question, df):
    followup_questions = generate_followup_cached(question=question, df=df)
    result = []
    
    if len(followup_questions) > 0:
        for followup_question in followup_questions[:5]:
            time.sleep(0.05)
            result.append(followup_question)

    return result   


def create_llms_model():
    llm = CTransformers(model="mistral-7b-instruct-v0.1.Q2_K.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm




#@st.cache_resource
def qa_llm():
    llm = create_llms_model()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 3})
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
memory = ConversationBufferWindowMemory(memory_key="chat_history",k=3, return_messages=True)



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

followup_result = []  # Declare followup_result here

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Ask your Query here", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = process_and_display_sql(user_input)
        followup_result = followup_questions(user_input, output)

        # Display chat history
        if isinstance(output, str):
            st.warning("Query execution error. Switching to another engine.")
            output = conversation_chat(user_input)
            #follow = generate_questions_cached()

        # Update session state
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

        # Display follow-up questions
        if followup_result:
            st.write("Follow-up Questions:")
            for followup_question in followup_result:
                st.write(followup_question)

# Display chat history
if st.session_state['generated']:
    with reply_container:
        for i, generated_output in enumerate(st.session_state['generated']):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")

            # Display output in chat
            if isinstance(generated_output, pd.DataFrame):
                assistant_message_table = st.chat_message("assistant")
                if len(generated_output) > 10:
                    assistant_message_table.text("First 10 rows of data")
                    assistant_message_table.dataframe(generated_output.head(10))
                else:
                    assistant_message_table.dataframe(generated_output)
                
                # Display follow-up questions below the answer
                if followup_result:
                    assistant_message_table.write("Follow-up Questions:")
                    for followup_question in followup_result:
                        assistant_message_table.write(followup_question)
            else:
                # Display non-DataFrame output
                message(generated_output, key=str(i), avatar_style="fun-emoji")
                

# ...

st.write(memory.load_memory_variables({}))          