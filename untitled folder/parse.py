import time
import os
from llmware.library import Library
from llmware.retrieval import Query
import tabula
import jpype
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
import pandas as pd
from openai import OpenAI
from functools import reduce

'''
This section is the most important part
1) make sure that you are running this file on the same MongoDB port
2) add seperate input and output folders for each pdf
3) establish open_ai_key
'''

amazon_folder = '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/input_folder'
amazon_output = '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/amazon_folder'

disney_folder = '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_folder'
disney_output = '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output'

solutions_folder = '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/solutions_folder'
solutions_output = '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/solutions_output'

openai_api_key = ''


'''
llmware is another RAG framework
1) essentially you create a new library everytime to become structured in json / csv 
2) specific to query to the company

'''


def parsing_with_fin_docs(input_fp, output_fp, query, newLibrary):
    t0 = time.time()
    lib = Library().create_new_library(newLibrary)
    parsing_output = lib.add_files(input_folder_path=input_fp)
    print("time to parse: ", time.time() - t0)
    print("parsing output: ", parsing_output)

    # structured data
    outputOne = lib.export_library_to_jsonl_file(output_fp, "info.jsonl")
    outputTwo = Query(lib).export_all_tables(query=query, output_fp=output_fp)

    return 0

amazon = parsing_with_fin_docs(amazon_folder, amazon_output, "amazon", 'newlibraryOne')
solutions = parsing_with_fin_docs(solutions_folder, solutions_output, "pdf solutions", 'newlibraryTwo')
disney = parsing_with_fin_docs(disney_folder, disney_output, "find all relevant data from disney and disney's finances tables from the inputed folder", 'newlibraryThree')


# recognize that csv for Walt Disney and PDF Solutions aren't able to added directly, so use tabular to add manually

def add_csv(pdf, folder_path):
    tables = tabula.read_pdf(pdf, pages='all', multiple_tables=True)

    formatted_dfs = []

    # Iterate over each table in the 'tables' list
    for table in tables:
        # Remove the first column if it contains only NaNs
        if table.iloc[:, 0].isnull().all():
            table = table.iloc[:, 1:]
        
        # Remove rows with all NaN values
        table = table.dropna(how='all')
        
        # Reset the index of the DataFrame
        table = table.reset_index(drop=True)
        
        # Append the formatted DataFrame to the list
        formatted_dfs.append(table)
    
    for i, df in enumerate(formatted_dfs, start=1):
        csv_filename = os.path.join(folder_path, f"dataframe_{i}.csv")
        df.to_csv(csv_filename, index=False)

add_csv('/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_folder/q1-fy24-earnings.pdf', disney_output)
add_csv('solutions_folder/Management Report Q1 2024 PDFS copy.pdf', solutions_output)



# at this point, all the parsing is done, and now needs to be condensed into a structured form

disneyAgent = create_csv_agent(
    ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo-0613"),
    ['/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_1.csv', 
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_2.csv',
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_3.csv',
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_4.csv',
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_5.csv',
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_6.csv',
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_7.csv',
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_8.csv',
    '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/disney_output/dataframe_9.csv'],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

solutionAgent = create_csv_agent(
    ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo-0613"),
    ['/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/solutions_output/dataframe_1.csv', '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/solutions_output/dataframe_2.csv', '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/solutions_output/dataframe_3.csv', '/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/solutions_output/dataframe_4.csv' ],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)


# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def read_csv_file(file_path):
    """Reads a CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path, sep='delimiter', header=None, engine='python')
    return df

def amazon_call(question, data, model="gpt-3.5-turbo"):
    """Makes a call to OpenAI with the provided question and data."""
    data_str = data.to_string(index=False)
    
    chat_completion = client.chat.completions.create(messages=[
        {
            "role": "user",
            "content": f"Answer this question: {question}, based on the input that you are given {data_str}. And be very specific in answering questions, you are a detailed financial analyst."
        }],
        model=model,
    )
    response_1 = chat_completion.choices[0].message.content
    return response_1

def process_and_merge_csv_files(file_paths, key_column='key'):
    """Reads multiple CSV files, processes them, and merges them on a key column."""
    array = [read_csv_file(file_path) for file_path in file_paths]
    
    for i, df in enumerate(array):
        if key_column not in df.columns:
            df[key_column] = range(len(df))  # Generate the key column if not present
        array[i] = df
    
    # Merge the DataFrames on the key column
    result = reduce(lambda left, right: pd.merge(left, right, on=key_column), array)
    return result

def amazonAgent(question):
    """Main function to read data, merge it, and ask a question using OpenAI."""
    file_paths = [f'/Users/aryanpanda/Downloads/digit-classifier-main/untitled folder/amazon_folder/table_{i}.csv' for i in range(0,3)]
    merged_data = process_and_merge_csv_files(file_paths)
    
    response = amazon_call(question, merged_data)
    
    print(response)



