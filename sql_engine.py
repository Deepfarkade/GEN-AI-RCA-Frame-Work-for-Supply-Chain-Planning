import time
import os
from code_editor import code_editor
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

# You can use the setup_session_state function here if needed.

def set_question(question):
    session_state["my_question"] = question

def show_suggested_questions():
    session_state["my_question"] = None
    questions = generate_questions_cached()
    for i, question in enumerate(questions):
        time.sleep(0.05)
        print(f"{i+1}. {question}")
        input("Press Enter to continue...")
        set_question(question)

if __name__ == "__main__":
    print("Happy Birthday Srikanth")

    my_question = input("Ask me a question about your data: ")

    session_state = {}

    if my_question.lower() == "suggested":
        show_suggested_questions()

    if my_question:
        session_state["my_question"] = my_question
        print(f"User: {my_question}")

        sql = generate_sql_cached(question=my_question)

        if sql:
            print(f"Generated SQL code:\n{sql}")
            if sql != "":
                df = run_sql_cached(sql=sql)
            else:
                df = None    

            #happy_sql = input("Are you happy with the generated SQL code? (yes/no): ")

            #if happy_sql.lower() == "no":
                #print("Please fix the generated SQL code.")
                #sql_response = code_editor(sql, lang="sql")
                #fixed_sql_query = sql_response["text"]

                #if fixed_sql_query != "":
                    #df = run_sql_cached(sql=fixed_sql_query)
                #else:
                    #df = None
            #elif happy_sql.lower() == "yes":
                #df = run_sql_cached(sql=sql)
            #else:
                #df = None

            if df is not None:
                session_state["df"] = df

                if session_state.get("df") is not None:
                    df = session_state.get("df")
                    if len(df) > 10:
                        print("First 10 rows of data:")
                        print(df.head(10))
                    else:
                        print(df)

                    code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                    ##print("Generated Plotly code:")
                    #print(code)

                    #happy_plotly = input("Are you happy with the generated Plotly code? (yes/no): ")

                    #if happy_plotly.lower() == "no":
                        #print("Please fix the generated Python code.")
                        #python_code_response = code_editor(code, lang="python")
                        #code = python_code_response["text"]
                    #elif happy_plotly.lower() == "":
                        #code = None

                    if code is not None and code != "":
                        print("Displaying the generated chart:")
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            #print(fig)
                            print("Image Generated")
                        else:
                            print("I couldn't generate a chart")


                        followup_questions = generate_followup_cached(question=my_question, df=df)

                        if len(followup_questions) > 0:
                            print("Here are some possible follow-up questions:")
                            for question in followup_questions[:5]:
                                time.sleep(0.05)
                                print(question)

                else:
                    print("Dataframe is None.")

            else:
                print("Dataframe is None.")

        else:
            print("I wasn't able to generate SQL for that question")

