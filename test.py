import streamlit as st
import os
import json
import pandas as pd
import spacy
from streamlit_chat import message
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from utils.setup import setup_connexion, setup_session_state
from utils.vanna_calls import generate_sql_cached, run_sql_cached

os.environ['REQUESTS_CA_BUNDLE'] = 'cert.crt'

setup_connexion()

TF_ENABLE_ONEDNN_OPTS = 0




# Load spaCy model
nlp = 1 #spacy.load("en_core_web_lg")

# Define path to the predefined questions JSON file
#PREDEFINED_QUESTIONS_FILE = "predefined_questions.json"

def find_similar_question(query, user_role):
    global PREDEFINED_QUESTIONS_FILE  # Declare the variable as global

    max_similarity = 0
    similar_question = None
    
    # Define the file path based on the user's role
    if user_role == "PPC User":
        predefined_questions_file = "PPC_Predefined_questions.json"
    elif user_role == "Coated User":
        predefined_questions_file = "Coated_Predefined_questions.json"
    else:
        # Default to a general predefined questions file
        predefined_questions_file = PREDEFINED_QUESTIONS_FILE

    # Load predefined questions from JSON file
    with open(predefined_questions_file, 'r') as f:
        predefined_questions = json.load(f)

    # Iterate through predefined questions to find the most similar one
    for question in predefined_questions:
        similarity = nlp(question).similarity(nlp(query))
        if similarity > max_similarity:
            max_similarity = similarity
            similar_question = question
    
    return similar_question

def prompt_next_question(query, user_role):
    similar_question = find_similar_question(query, user_role)
    if similar_question:
        # Define the file path based on the user's role
        if user_role == "PPC User":
            predefined_questions_file = "PPC_Predefined_questions.json"
        elif user_role == "Coated User":
            predefined_questions_file = "Coated_Predefined_questions.json"
        else:
            # Default to the general predefined questions file
            predefined_questions_file = PREDEFINED_QUESTIONS_FILE
        
        # Load predefined questions from the appropriate JSON file
        with open(predefined_questions_file, 'r') as f:
            predefined_questions = json.load(f)
        
        # Find the index of the similar question
        index = predefined_questions.index(similar_question)
        
        # Get the next question in the sequence
        if index + 1 < len(predefined_questions):
            next_question = predefined_questions[index + 1]
        else:
            # Loop back to the first question if the prompted question is the last question
            next_question = predefined_questions[0]
        
        return next_question
    return None

def process_and_display_sql(question, user_role):
    sql = generate_sql_cached(question=question)
    if sql:
        # Process SQL query
        if sql != "":
            df = run_sql_cached(sql=sql)
            output = "SQL Engine Result: " + str(df)

            # Save to memory
            memory.save_context({"input": question}, {"output": output})

            # Save user's question to user role-specific file
            save_user_question(question, user_role)

            return df
    else:
        # If no SQL query is generated, you may want to handle this case accordingly
        return "Sorry, I couldn't generate a valid SQL query for your question."

def save_user_question(question, user_role):
    user_role_file_path = f"{user_role.lower().replace(' ', '_')}_questions.json"

    # Load existing questions from file if it exists
    if os.path.exists(user_role_file_path):
        with open(user_role_file_path, "r") as f:
            user_questions = json.load(f)
    else:
        user_questions = {}

    # Update count or add new question
    user_questions[question] = user_questions.get(question, 0) + 1

    # Save updated questions to file
    with open(user_role_file_path, "w") as f:
        json.dump(user_questions, f, indent=2)  # Write dictionary with indentation for readability

def load_user_questions(user_role):
    user_role_file_path = f"{user_role.lower().replace(' ', '_')}_questions.json"

    if os.path.exists(user_role_file_path):
        with open(user_role_file_path, "r") as f:
            return json.load(f)
    else:
        return {}  # Return an empty dictionary if file doesn't exist

def get_top_5_questions(user_role):
    user_questions = load_user_questions(user_role)

    # Sort questions by count in descending order
    sorted_questions = sorted(user_questions.items(), key=lambda x: x[1], reverse=True)

    # Return the top 5 questions
    top_5_questions = [question for question, count in sorted_questions[:5]]
    return top_5_questions

def create_llms_model():
    llm = 1 # CTransformers(model="mistral-7b-instruct-v0.1.Q2_K.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm

def qa_llm():
    llm = create_llms_model()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory=memory)
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

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


# Initialize Streamlit app
st.title("Gen AI Chatbot")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)

# Check for logged_out query parameter to render login page
if st.experimental_get_query_params().get("logged_out"):
    st.experimental_set_query_params()  # Clear query parameters
    st.session_state.page_selection = "login"


# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Dynamic initial message based on logged-in user's name
logged_in_user = st.session_state.get('username', '')
initial_message = f"Hello, {logged_in_user}! Please ask me your query. 🤗"

# Ensure 'generated' session state is properly initialized
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# Append initial message if the 'generated' session state is empty
if not st.session_state['generated']:
    st.session_state['generated'].append(initial_message)   

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]
    
# Ensure 'username' session state is properly initialized
if 'username' not in st.session_state:
    st.session_state['username'] = ''    

# Create memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4, return_messages=True)

# Check if users.json file exists, if not, create an empty one
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

# Load existing users from the JSON file
try:
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
except json.JSONDecodeError:
    users = {}

# Separate page for login and signup
page = st.empty()

if "page_selection" not in st.session_state:
    st.session_state.page_selection = "login"

if st.session_state.page_selection == "login":
    login_form = st.form(key='login_form')
    with login_form:
        st.header("Login")
        username_login = st.text_input("Username", key='username_login')
        password_login = st.text_input("Password", type='password', key='password_login')
        login_button = st.form_submit_button("Login")

    if login_button:
        if username_login in users and users[username_login]["password"] == password_login:
            st.success("Login Successful!")
            st.session_state['logged_in'] = True
            st.session_state['username'] = username_login
            st.session_state['user_role'] = users[username_login]["user_role"]
            #st.session_state['user_role'] = users[username_login]["user_role"]  # Set user_role in session state
            st.session_state.page_selection = "chatbot"
        else:
            st.error("Invalid username or password. Please try again.")

    # Show signup button even if login is unsuccessful
    signup_button = st.button("Create Account")
    if signup_button:
        st.session_state.page_selection = "signup"

elif st.session_state.page_selection == "signup":
    signup_form = st.form(key='signup_form')
    with signup_form:
        st.header("Sign Up")
        username_signup = st.text_input("Username", key='username_signup')
        password_signup = st.text_input("Password", type='password', key='password_signup')
        user_role_signup = st.selectbox("Select User Role", ["Coated User", "PPC User"], key='user_role_signup')
        signup_button = st.form_submit_button("Sign Up")

    if signup_button:
        if username_signup not in users:
            users[username_signup] = {"password": password_signup, "user_role": user_role_signup}
            with open(USERS_FILE, "w") as f:
                json.dump(users, f)
            st.success("Sign Up Successful! You can now log in.")
            st.session_state.page_selection = "login"

    # Back to Login button
    back_to_login_button = st.button("Back to Login")
    if back_to_login_button:
        st.session_state.page_selection = "login"

elif st.session_state.page_selection == "chatbot":
    # Display chat history
    reply_container = st.container()
    container = st.container()


    if 'prompts_displayed' not in st.session_state:
        st.session_state['prompts_displayed'] = False
        
    with container:
        #if not st.session_state.get('prompts_displayed', False):
        #if len(st.session_state['generated']) == 1 and len(st.session_state['past']) == 1:  # Only display top 5 questions if no ongoing conversation
        if not st.session_state['prompts_displayed']:
            # Load user-specific questions for prompts
            user_role = st.session_state.get('user_role', '')  # Assuming user_role is stored in the session state
            top_5_questions = get_top_5_questions(user_role)

            # Display prompts
            if top_5_questions:
                st.write("You can also ask:")
                for i, question in enumerate(top_5_questions, start=1):
                    st.write(f"{i}. {question}")

            # Mark prompts as displayed
            st.session_state['prompts_displayed'] = True
            
            
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask your Query here", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            user_role = st.session_state.get('user_role', '')  # Access user_role from session state
            # Process and display user's query
            output = process_and_display_sql(user_input, st.session_state.get('user_role', ''))  # Passing user_role as an argument
            
            # Display chat history
            if isinstance(output, str):
                #st.warning("Give me some time I am thinking more in deeper way on your question. 🤗")
                st.warning("Can you please Reframe / Elaborate your Question. 🤗")
                #output = conversation_chat(user_input)
            # Update session state
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            

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
                        
                        # Check if user_role exists before calling prompt_next_question
                        if 'user_role' in st.session_state:
                            next_question = prompt_next_question(user_input, st.session_state['user_role'])
                        else:
                            # Handle the case when user_role is not defined
                            next_question = None  # or any other appropriate action
                        
                        #next_question = prompt_next_question(user_input, user_role)
                        with st.chat_message("assistant"):
                            st.write(f"I think you would love to ask :\n {next_question}")
                    else:
                        assistant_message_table.dataframe(generated_output)
                        # Check if user_role exists before calling prompt_next_question
                        if 'user_role' in st.session_state:
                            next_question = prompt_next_question(user_input, st.session_state['user_role'])
                        else:
                            # Handle the case when user_role is not defined
                            next_question = None  # or any other appropriate action
                        
                        #next_question = prompt_next_question(user_input, user_role)
                        with st.chat_message("assistant"):
                            st.write(f"I think you would love to ask :\n {next_question}")

                else:
                    # Display non-DataFrame output
                    message(generated_output, key=str(i), avatar_style="fun-emoji")
                    #next_question = prompt_next_question(user_input)
                    #with st.chat_message("assistant"):
                        #st.write(f"I think you would love to ask: {next_question}")

    #st.write(memory.load_memory_variables({}))
    # Logout button
    logout_button = st.button("Logout", key='chatbot_logout_button')
    if logout_button:
        # Clear session_state variables related to chat history
        if 'history' in st.session_state:
            del st.session_state['history']

        if 'generated' in st.session_state:
            del st.session_state['generated']

        if 'past' in st.session_state:
            del st.session_state['past']
        
        # Redirect to login page and clear query parameters
        st.experimental_set_query_params(logged_out=True)
            

else:
    st.warning("Invalid page selection.")