import streamlit as st
import os
from utils.setup import setup_connexion, setup_session_state
from test2 import memory
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
#st.set_page_config(layout="wide")

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
                        
# Example usage
question = "What is the average sales per month?"
process_and_display_sql(question)                