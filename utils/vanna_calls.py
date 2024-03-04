import streamlit as st
import vanna 
import os
from sql_execution import execute_sql_query
from vanna.remote import VannaDefault

os.environ['REQUESTS_CA_BUNDLE'] = 'cert.crt'

api_key = "2e8f6ca84f514b6eab6250c51f6a6d93"
vanna_model_name = "esp_model"
vn = VannaDefault(model=vanna_model_name, api_key=api_key)


@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    return vn.generate_questions()


#@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    return vn.generate_sql(question=question)


#@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    #return vn.run_sql(sql=sql)
    return execute_sql_query(sql=sql)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, df):
    return vn.generate_followup_questions(question=question, df=df)
