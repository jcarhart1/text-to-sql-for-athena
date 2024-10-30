import logging    # Allows the program to log messages, which can help in debugging and tracking events
import json       # Enables the program to work with JSON data formats
import os         # Provides a way to interact with the operating system
import sys        # Gives access to variables and functions that interact with the Python interpreter
import re         # Supports regular expression matching operations
# sys.path.append("/home/ec2-user/SageMaker/llm_bedrock_v0/")  # (Commented out) Would add a specific directory to the system path for module access
from boto_client import Clientmodules  # Imports custom client modules to interact with AWS services
import time       # Provides time-related functions, such as delays
import pandas as pd   # Imports pandas library, used for data manipulation and analysis
import io         # Used to handle input/output operations, especially in-memory streams

# Sets up a logger to record events and errors, which is helpful for monitoring the application's behavior
logger = logging.getLogger(__name__)   # Creates a logger with the name of the current module
logger.setLevel(logging.DEBUG)         # Sets the logging level to DEBUG to capture detailed information
logger.addHandler(logging.StreamHandler())  # Adds a handler to output log messages to the console

# Defines a class to encapsulate Athena query execution functionality to interact with Athena and S3
class AthenaQueryExecute:
    def __init__(self):
        # Initializes the class and sets up necessary configurations and clients

        # Specifies the name of the S3 bucket where Athena will store query results
        self.glue_databucket_name = 'athena-query-results-text-to-sql'

        # Creates a client to interact with Amazon Athena using a custom module
        self.athena_client = Clientmodules.createAthenaClient()

        # Creates a client to interact with Amazon S3 using a custom module
        self.s3_client = Clientmodules.createS3Client()

    # Defines a method to execute a given SQL query in Athena
    def execute_query(self, query_string):
        # Specifies the folder in S3 where the query results will be stored
        result_folder = 'athena_output'

        # Configures where Athena should output the query results in S3
        result_config = {"OutputLocation": f"s3://{self.glue_databucket_name}/{result_folder}"}

        # Sets the context for the query execution, such as the data catalog
        query_execution_context = {
            "Catalog": "AwsDataCatalog",
        }

        # Prints the SQL query that is about to be executed
        print(f"Executing: {query_string}")

        # Initiates the execution of the SQL query in Athena
        query_execution = self.athena_client.start_query_execution(
            QueryString=query_string,                # The SQL query to execute
            ResultConfiguration=result_config,       # Where to store the results in S3
            QueryExecutionContext=query_execution_context,  # Additional execution context
        )

        # Retrieves the unique ID assigned to this query execution
        execution_id = query_execution["QueryExecutionId"]

        # Waits for a specified amount of time to allow the query to complete
        time.sleep(120)

        # Constructs the filename for the query results based on the execution ID
        file_name = f"{result_folder}/{execution_id}.csv"
        logger.info(f'Checking for file: {file_name}')  # Logs the filename being checked

        # Specifies the local path where the query results will be temporarily stored
        local_file_name = f"./tmp/{file_name}"

        # Prints the parameters that will be used to download the file from S3
        print(f"Calling download file with params {local_file_name}, {result_config}")

        # Retrieves the query results file from S3
        obj = self.s3_client.get_object(Bucket=self.glue_databucket_name, Key=file_name)

        # Reads the CSV data from the S3 object into a pandas DataFrame for easy data manipulation
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')

        # Returns the DataFrame containing the query results
        return df

    # Defines a method to check the syntax of a SQL query before execution
    def syntax_checker(self, query_string):
        # Prints a message indicating that syntax checking has started
        print("Inside syntax_checker", query_string)

        # Specifies the folder in S3 where the syntax check results will be stored
        query_result_folder = 'athena_query_output/'

        # Configures where Athena should output the syntax check results in S3
        query_config = {"OutputLocation": f"s3://{self.glue_databucket_name}/{query_result_folder}"}

        # Sets the context for the query execution
        query_execution_context = {
            "Catalog": "AwsDataCatalog",
        }

        # Modifies the query to include the 'EXPLAIN' command, which checks the query without running it
        query_string = "Explain  " + query_string

        # Prints the modified query being executed for syntax checking
        print(f"Executing: {query_string}")

        try:
            # Indicates that the program is about to check the syntax
            print("I am checking the syntax here")

            # Initiates the execution of the syntax check in Athena
            query_execution = self.athena_client.start_query_execution(
                QueryString=query_string,                # The modified SQL query for syntax checking
                ResultConfiguration=query_config,        # Where to store the syntax check results in S3
                QueryExecutionContext=query_execution_context,  # Additional execution context
            )

            # Retrieves the unique ID assigned to this syntax check execution
            execution_id = query_execution["QueryExecutionId"]
            print(f"execution_id: {execution_id}")  # Prints the execution ID for reference

            # Waits briefly to allow the syntax check to complete
            time.sleep(3)

            # Retrieves the execution status of the syntax check
            results = self.athena_client.get_query_execution(QueryExecutionId=execution_id)
            status = results['QueryExecution']['Status']
            print("Status:", status)  # Prints the status of the syntax check

            # Checks if the syntax check succeeded
            if status['State'] == 'SUCCEEDED':
                return "Passed"  # Returns "Passed" if the syntax is correct
            else:
                # Prints and returns the reason why the syntax check failed
                print(results['QueryExecution']['Status']['StateChangeReason'])
                errmsg = results['QueryExecution']['Status']['StateChangeReason']
                return errmsg
        except Exception as e:
            # Handles any exceptions that occur during the syntax check
            print("Error in exception")
            msg = str(e)
            print(msg)  # Prints the exception message for debugging
