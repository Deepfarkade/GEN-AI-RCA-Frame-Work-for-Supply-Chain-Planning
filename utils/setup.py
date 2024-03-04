import os
import streamlit as st
import vanna as vn
from dotenv import load_dotenv
from vanna.remote import VannaDefault


api_key = "2e8f6ca84f514b6eab6250c51f6a6d93"

vanna_model_name = "esp_model"
vn = VannaDefault(model=vanna_model_name, api_key=api_key)


def setup_session_state():
    st.session_state["my_question"] = None
