from sqlalchemy import create_engine
import pandas as pd
from app_secrets import *


def execute_sql_query(sql):
    # SQL Server connection parameters
    #connection_params = {
    #    'mssql+pyodbc': f"DRIVER={ODBC_DRIVER};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};Trusted_Connection=yes"
    #}
    
    # SQL Server connection parameters
    connection_params = {
    'mssql+pyodbc': f"mssql+pyodbc:///?odbc_connect=DRIVER={ODBC_DRIVER};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};Trusted_Connection=yes"
    }

    query = sql

    try:
        # Establish a connection to SQL Server
        engine = create_engine(connection_params['mssql+pyodbc'])

        # Execute the query
        try:
            query_results = pd.read_sql_query(query, engine)
        except Exception as e:
            print("Query Execution Error:", e)
            return "Query execution error"

        # Print the DataFrame
        # print(query_results)
        return query_results

    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close the engine
        try:
            engine.dispose()
        except:
            pass

if __name__ == "__main__":
    # SQL Server query
    query = '''
            SELECT * FROM [ITEM]
    '''
    execute_sql_query(query)
