import pandas as pd
import io

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI, OpenAI
import os
import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import io
import csv

def read_csv_file(csv_file):
    """Reads a CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_file, sep='delimiter', header=None, engine='python')
    return df


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    data = read_csv_file(csv_file)

    if csv_file is not None:
        agent = create_pandas_dataframe_agent(
    ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0, model="gpt-3.5-turbo-0613"), data, verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.invoke(user_question))


if __name__ == "__main__":
    main()