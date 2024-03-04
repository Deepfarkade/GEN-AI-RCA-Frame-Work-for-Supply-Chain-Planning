import os
import streamlit as st
import vanna
from dotenv import load_dotenv
from vanna.remote import VannaDefault


@st.cache_resource(ttl=3600)
def setup_connexion():
    if "vanna_api_key" in st.secrets and "gcp_project_id" in st.secrets:
        api_key = "2e8f6ca84f514b6eab6250c51f6a6d93"
        vanna_model_name = "esp_model"
        vn = VannaDefault(model=vanna_model_name, api_key=api_key)
    else:
        load_dotenv()
        api_key = os.environ.get("VANNA_API_KEY")
        vanna_model_name = "esp_model"
        vn = VannaDefault(model=vanna_model_name, api_key=api_key)
        

#api_key = # Your API key from https://vanna.ai/account/profile 

#vanna_model_name = # Your model name from https://vanna.ai/account/profile 
#vn = VannaDefault(model=vanna_model_name, api_key=api_key)

def setup_session_state():
    st.session_state["my_question"] = None
