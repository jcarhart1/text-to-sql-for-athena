# designed to automate the process of generating and executing SQL queries based on natural language user input.
# It integrates several AWS services, including Amazon Bedrock for language modeling, Amazon Athena for query
# execution, and Amazon OpenSearch Service for vector-based similarity searches. The script takes a user's question,
# searches for relevant context using OpenSearch embeddings, constructs a prompt for the language model to generate a
# SQL query, checks the syntax of the generated SQL, and finally executes the query in Athena to retrieve results.
# This automation facilitates converting human language queries into executable SQL statements, enabling users to
# interact with databases without deep knowledge of SQL.


import boto3  # AWS SDK for Python to interact with AWS services
import logging  # Enables logging for debugging and monitoring
from boto_client import Clientmodules  # Custom module to create AWS service clients
from llm_basemodel import LanguageModel  # Custom class for language model initialization
from athena_execution import AthenaQueryExecute  # Custom class for executing Athena queries
from openSearchVCEmbedding import EmbeddingBedrockOpenSearch  # Custom class for OpenSearch operations
from langchain.schema import AIMessage  # Schema for handling AI model messages

# Initializes a new AWS session
session = boto3.session.Session()
# Creates a client for Amazon Bedrock service
bedrock_client = session.client('bedrock')
# Prints the first available foundation model summary from Bedrock
print(bedrock_client.list_foundation_models()['modelSummaries'][0])

# Instantiates the AthenaQueryExecute class to handle Athena query execution
rqstath = AthenaQueryExecute()

# Sets up logging configuration
logger = logging.getLogger(__name__)  # Creates a logger with the name of the current module
logger.setLevel(logging.DEBUG)  # Sets the logging level to DEBUG
logger.addHandler(logging.StreamHandler())  # Adds a handler to output log messages to the console

# **Configuration Parameters for OpenSearch**
index_name = 'text_to_sql_index'  # Name of the OpenSearch index
domain = 'https://e0hc00i67ga6mpn1xkxa.us-east-1.aoss.amazonaws.com'  # OpenSearch domain endpoint
region = 'us-east-1'  # AWS region
vector_name = 'embeddings_vector'  # Name of the vector field in the index
fieldname = 'id'  # Name of the text field in the index

# Creates an instance of EmbeddingBedrockOpenSearch for OpenSearch operations
ebropen2 = EmbeddingBedrockOpenSearch(domain, vector_name, fieldname)
# Checks if the ebropen2 object is created successfully
if ebropen2 is None:
    print("ebropen2 is null")
else:
    # Prints the attributes of the ebropen2 object for debugging purposes
    attrs = vars(ebropen2)
    print(', '.join("%s: %s" % item for item in attrs.items()))


# **Class Definition**

# Defines a class to handle the request and query processing using Bedrock
class RequestQueryBedrock:
    def __init__(self, ebropen2):
        self.ebropen2 = ebropen2  # Stores the OpenSearch embedding object
        self.bedrock_client = ebropen2.bedrock_client  # Retrieves the Bedrock client
        if self.bedrock_client is None:
            # Creates a new Bedrock runtime client if not available
            self.bedrock_client = Clientmodules.createBedrockRuntimeClient()
        else:
            print("the bedrock_client is not null")
        self.language_model = LanguageModel(self.bedrock_client)  # Initializes the language model
        self.llm = self.language_model.llm  # Assigns the language model instance

    # Retrieves context from OpenSearch embeddings based on the user query
    def getOpenSearchEmbedding(self, index_name, user_query):
        vcindxdoc = self.ebropen2.getDocumentfromIndex(index_name=index_name)  # Retrieves the index
        document = self.ebropen2.getSimilaritySearch(user_query, vcindxdoc)  # Performs similarity search
        return self.ebropen2.get_data(document)  # Extracts data from documents

    # Generates SQL query using the language model and handles retries
    def generate_sql(self, prompt, max_attempt=4) -> str:
        attempt = 0  # Initializes the attempt counter
        error_messages = []  # Stores error messages for reporting
        prompts = [prompt]  # List to keep track of prompts used
        sql_query = ""  # Placeholder for the generated SQL query

        while attempt < max_attempt:
            logger.info(f'Sql Generation attempt Count: {attempt + 1}')  # Logs the attempt number
            try:
                logger.info(f'we are in Try block to generate the sql and count is :{attempt + 1}')
                generated_sql = self.llm.invoke(prompt)  # Invokes the language model with the prompt

                # Handles the response depending on its type
                if isinstance(generated_sql, AIMessage):
                    content = generated_sql.content  # Extracts content from AIMessage
                elif isinstance(generated_sql, str):
                    content = generated_sql  # Uses the string as content
                else:
                    content = str(generated_sql)  # Converts other types to string

                # Extracts SQL query from the generated content
                sql_parts = content.split("```")  # Splits the content by code block delimiters
                if len(sql_parts) > 1:
                    query_str = sql_parts[1]  # Assumes SQL is within code blocks
                else:
                    query_str = content  # Uses the entire content if no code blocks

                query_str = " ".join(query_str.split("\n")).strip()  # Removes newlines and extra spaces
                # Removes 'SQL' prefix if present
                sql_query = query_str[3:] if query_str.lower().startswith("sql") else query_str

                print(sql_query)  # Prints the generated SQL query
                # Checks the syntax of the generated SQL query using Athena
                syntaxcheckmsg = rqstath.syntax_checker(sql_query)
                if syntaxcheckmsg == 'Passed':
                    logger.info(f'syntax checked for query passed in attempt number :{attempt + 1}')
                    return sql_query  # Returns the valid SQL query
                else:
                    # Prepares a new prompt incorporating the syntax error message
                    prompt = f"""{prompt}
                            This is syntax error: {syntaxcheckmsg}. 
                            To correct this, please generate an alternative SQL query which will correct the syntax error.
                            The updated query should take care of all the syntax issues encountered.
                            Follow the instructions mentioned above to remediate the error. 
                            Update the below SQL query to resolve the issue:
                            {sql_query}
                            Make sure the updated SQL query aligns with the requirements provided in the initial question."""
                    prompts.append(prompt)  # Adds the new prompt to the list
            except Exception as e:
                print(e)  # Prints the exception message
                logger.error('FAILED')  # Logs the failure
                msg = str(e)
                error_messages.append(msg)  # Adds the error message to the list
            finally:
                attempt += 1  # Increments the attempt counter

        # Raises an exception if all attempts fail, including error details
        raise Exception(f"Failed to generate SQL after {max_attempt} attempts. Errors: {', '.join(error_messages)}")


# Creates an instance of the RequestQueryBedrock class
rqst = RequestQueryBedrock(ebropen2)


# **Function Definitions**

# Handles user input and orchestrates the process of generating SQL queries
def userinput(user_query):
    logger.info(f'Searching metadata from vector store')
    # Retrieves relevant context using OpenSearch embeddings
    vector_search_match = rqst.getOpenSearchEmbedding(index_name, user_query)

    # Detailed instructions for the language model to generate SQL queries
    details = f"""It is important that the SQL query complies with Athena syntax.

                  For our testing purposes, please use ONLY the following data source(s), database(s) and table(s):

                  Data Source: AWSDataCatalog              
                  Database: imdb_stg
                  Tables: basics, ratings
                  Columns: Use only the columns present in the tables mentioned above
                  Unique ID: tconst

                  Think of these parameters as the only available resources with which you have to answer the question. 

                  During a join, if two column names are the same please use alias (example: basics.tconst in select 
                  statement). It is also important to pay attention to and not alter column format: if a column is string, 
                  then leave column formatting alone and return a value that is a string. 

                  If you are writing CTEs then include all the required columns. While concatenating a non-string column, 
                  make sure to cast the column to string format first. If you encounter any instances where we must 
                  compare date columns to strings, please cast the string input as a date and format as such. 

                  REMEMBER: Only use the data source(s), database(s), and table(s) mentioned above. In addition,
                  always include the database name along with the table name in the query."""

    # Constructs the final prompt for the language model
    final_question = "\n\nHuman:" + details + vector_search_match + user_query + "\n\nAssistant:"
    print("FINAL QUESTION :::" + final_question)

    try:
        # Generates the SQL query using the prepared prompt
        answer = rqst.generate_sql(final_question)
        return answer
    except Exception as e:
        logger.error(f"Failed to generate SQL: {str(e)}")
        return None  # Returns None if generation fails


# **Main Execution**

# Defines the main function to run when the script is executed
def main():
    # Example user queries (only one is active at a time)
    # user_query = 'How many records in our database are from the year 1892?'
    user_query = 'What was the total number of votes for all movies with the word clown in the title?'
    # user_query = 'I need all of the unique ids from the Animation genre with an average rating of 5 or higher and at least 1000 votes'
    # Calls the userinput function with the user query
    querygenerated = userinput(user_query)

    if querygenerated:
        import pprint  # Imports pprint for pretty-printing
        my_printer = pprint.PrettyPrinter()
        my_printer.pprint(querygenerated)  # Prints the generated SQL query in a readable format

        # Executes the generated SQL query using Athena
        QueryOutput = rqstath.execute_query(querygenerated)
        print(QueryOutput)  # Prints the query results
    else:
        print("Failed to generate a valid SQL query.")  # Informs the user if query generation failed


# Checks if the script is being run directly
if __name__ == '__main__':
    main()  # Calls the main function to execute the script
