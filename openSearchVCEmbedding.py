
# This program represents a comprehensive solution for integrating language model embeddings with OpenSearch to perform
# semantic searches. It leverages AWS services like Bedrock for generating embeddings and OpenSearch for indexing and
# searching documents. The script defines a class that encapsulates the functionality needed to check index existence,
# add documents, retrieve the index, perform similarity searches, and process the results. The main function
# demonstrates how these methods work together to fulfill a user query.


import sys              # Provides access to system-specific parameters and functions
import time             # Provides time-related functions
import traceback        # Allows extraction and printing of stack traces for exception handling

from requests_aws4auth import AWS4Auth                   # Handles AWS authentication for requests
from opensearchpy import OpenSearch, RequestsHttpConnection  # Libraries for interacting with OpenSearch

from typing import List, Tuple    # Provides type hinting for better code readability
import logging                    # Enables logging of messages for debugging and monitoring
import numpy as np                # Imports NumPy library for numerical operations
import boto3                      # AWS SDK for Python to interact with AWS services

# Commented out imports that might be used elsewhere but are not active here
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import OpenSearchVectorSearch  # For vector-based OpenSearch operations
from langchain_community.document_loaders import JSONLoader          # For loading JSON documents

# Sets up a logger for the script
logger = logging.getLogger()  # Initializes the root logger
# Configures logging format and level (commented out here)
# logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO, stream=sys.stderr)

# Adds directories to the system path for module access (commented out)
# sys.path.append("/home/ec2-user/SageMaker/llm_bedrock_v0/")
# sys.path.append("//")

# Imports custom classes from local modules
from llm_basemodel import LanguageModel   # Class to handle language model initialization
from boto_client import Clientmodules     # Class to create AWS service clients

# Commented out import that might be used elsewhere
# from opensearchpy import AWSV4SignerAuth

# **Configuration Parameters**

# Specifies the OpenSearch domain endpoint
opensearch_domain_endpoint = 'https://e0hc00i67ga6mpn1xkxa.us-east-1.aoss.amazonaws.com'
aws_region = 'us-east-1'          # AWS region where services are hosted
index_name = 'text_to_sql_index'  # Name of the OpenSearch index to use
service = 'aoss'                  # AWS service name for OpenSearch
region = 'us-east-1'              # AWS region
credentials = boto3.Session().get_credentials()  # Retrieves AWS credentials for authentication

# Initializes AWS authentication for OpenSearch requests
awsauth = AWS4Auth(
    credentials.access_key,       # AWS access key ID
    credentials.secret_key,       # AWS secret access key
    region,                       # AWS region
    service,                      # AWS service name
    session_token=credentials.token  # AWS session token for temporary credentials
)
logger.info(awsauth)  # Logs the authentication details (may expose sensitive information)
i = 0  # Initializes a variable `i` to 0 (unused in this script)

# **Class Definition**

class EmbeddingBedrockOpenSearch:
    # Initializes the class with necessary parameters
    def __init__(self, domain, vector_name, fieldname):
        self.bedrock_client = Clientmodules.createBedrockRuntimeClient()  # Creates a Bedrock runtime client
        self.language_model = LanguageModel(self.bedrock_client)          # Initializes the language model
        print(self.language_model)                                        # Prints the language model details
        self.llm = self.language_model.llm                                # Assigns the language model instance
        self.embeddings = self.language_model.embeddings                  # Assigns the embeddings model instance
        self.opensearch_domain_endpoint = domain                          # Stores the OpenSearch domain endpoint
        self.http_auth = awsauth                                          # Stores the AWS authentication credentials
        self.vector_name = vector_name                                    # Name of the vector field in OpenSearch
        self.fieldname = fieldname                                        # Name of the text field in OpenSearch

        logger.info("created for domain " + domain)                       # Logs the domain creation message
        logger.info(credentials.access_key)                               # Logs the AWS access key (not recommended)

    # Checks if the specified index exists in OpenSearch
    def check_if_index_exists(self, index_name: str, region: str, host: str, http_auth: Tuple[str, str]) -> OpenSearch:
        hostname = host.replace("https://", "")                           # Removes 'https://' from the host URL

        logger.info(hostname)                                             # Logs the hostname
        aos_client = OpenSearch(                                          # Creates an OpenSearch client instance
            hosts=[{'host': hostname, 'port': 443}],                      # Specifies the host and port
            http_auth=awsauth,                                            # Uses AWS authentication
            use_ssl=True,                                                 # Enables SSL
            verify_certs=True,                                            # Verifies SSL certificates
            connection_class=RequestsHttpConnection,                      # Specifies the connection class
            timeout=300,                                                  # Sets the timeout for requests
            ssl_show_warn=True                                            # Shows SSL warnings
        )

        exists = aos_client.indices.exists(index_name)                    # Checks if the index exists
        print("exist check", exists)                                      # Prints the result of the check
        return exists                                                     # Returns the existence status (True/False)

    # Adds documents to the OpenSearch index
    def add_documents(self, index_name: str, file_name: str):
        # Loads documents from a JSON file using a JSONLoader
        documents = JSONLoader(
            file_path=file_name,    # Path to the JSON file
            jq_schema='.',          # JSON query schema to extract data
            text_content=False,     # Indicates that the content is not plain text
            json_lines=True         # Specifies that the JSON file has one JSON object per line
        ).load()

        # Ensures that metadata is in dictionary format to avoid validation errors
        for doc in documents:
            if isinstance(doc.metadata, str):
                doc.metadata = eval(doc.metadata)  # Converts string metadata to dictionary

        # Creates a vector search index in OpenSearch from the loaded documents
        docs = OpenSearchVectorSearch.from_documents(
            embedding=self.embeddings,                     # Embedding function to vectorize documents
            opensearch_url=self.opensearch_domain_endpoint,  # OpenSearch domain URL
            http_auth=self.http_auth,                      # AWS authentication credentials
            documents=documents,                           # Documents to be indexed
            index_name=index_name,                         # Name of the index to create/update
            engine="faiss"                                 # Specifies the vector engine (FAISS)
        )

        # Checks if the index now exists after attempting to add documents
        index_exists = self.check_if_index_exists(
            index_name,
            aws_region,
            self.opensearch_domain_endpoint,
            self.http_auth
        )
        logger.info(index_exists)                          # Logs whether the index exists
        print(index_exists)                                # Prints the existence status

        # If the index does not exist, exit the program with an error
        if not index_exists:
            logger.info(f'index :{index_name} is not existing ')
            sys.exit(-1)                                   # Exits the program with a status code of -1
        else:
            logger.info(f'index :{index_name} Got created')  # Logs that the index was successfully created

    # Retrieves the OpenSearch index for document search
    def getDocumentfromIndex(self, index_name: str):
        try:
            logger.info("the opensearch_url is " + self.opensearch_domain_endpoint)
            logger.info(self.http_auth)
            # Alternative authentication method (commented out)
            # http_auth = ('ll_vector','@')
            hostname = self.opensearch_domain_endpoint
            # Initializes the OpenSearch vector search client
            docsearch = OpenSearchVectorSearch(
                opensearch_url=hostname,                   # OpenSearch domain URL
                embedding_function=self.embeddings,        # Embedding function for query vectorization
                http_auth=self.http_auth,                  # AWS authentication credentials
                index_name=index_name,                     # Name of the index to search
                use_ssl=True,                              # Enables SSL
                connection_class=RequestsHttpConnection    # Specifies the connection class
            )

            return docsearch                               # Returns the OpenSearch vector search client
        except Exception:
            print(traceback.format_exc())                  # Prints the stack trace if an exception occurs

    # Performs a similarity search on the OpenSearch index
    def getSimilaritySearch(self, user_query: str, vcindex):
        # Example user query (commented out)
        # user_query='show me the top 10 titles by maximum votes'

        # Performs the similarity search using the vector index
        docs = vcindex.similarity_search(
            user_query,                     # The user's search query
            k=200,                          # Number of top documents to retrieve
            vector_field=self.vector_name,  # Name of the vector field in the index
            text_field=self.fieldname       # Name of the text field to retrieve
        )
        # print(docs[0].page_content)       # Prints the content of the first document (commented out)
        return docs                         # Returns the list of similar documents

    # Formats the metadata from the retrieved documents
    def format_metadata(self, metadata):
        docs = []
        # Removes indentation and line feeds from the text
        for elt in metadata:
            processed = elt.page_content    # Gets the content of the document
            print(processed)                # Prints the processed content
            # Example metadata access (commented out)
            # print (elt.metadata['x-amz-bedrock-kb-source-uri'])
            # print (elt.metadata['id'])

            chunk = elt.metadata['AMAZON_BEDROCK_TEXT_CHUNK']  # Retrieves a specific metadata field
            print(repr(chunk))                                 # Prints the raw representation of the chunk
            # Cleans up the text by removing extra spaces and newlines
            for i in range(20, -1, -1):
                processed = processed.replace('\n' + ' ' * i, '')

            docs.append(processed)          # Adds the cleaned text to the docs list
        result = '\n'.join(docs)            # Joins all documents into a single string
        # Escapes curly brackets to avoid formatting issues
        result = result.replace('{', '{{')
        result = result.replace('}', '}}')
        return result                       # Returns the formatted result

    # Alternative method to retrieve and clean data from metadata
    def get_data(self, metadata):
        docs = []
        # Removes indentation and line feeds from the text
        for elt in metadata:
            # processed = elt.page_content  # Commented out; not used in this method
            # print(processed)              # Commented out; would print the content
            # Example metadata access (commented out)
            # print (elt.metadata['x-amz-bedrock-kb-source-uri'])
            # print (elt.metadata['id'])

            chunk = elt.metadata['AMAZON_BEDROCK_TEXT_CHUNK']  # Retrieves a specific metadata field
            # Cleans up the text by removing extra spaces, newlines, and carriage returns
            for i in range(20, -1, -1):
                chunk = chunk.replace('\n' + ' ' * i, '')
                chunk = chunk.replace('\r' + ' ' * i, '')
            docs.append(chunk)              # Adds the cleaned text to the docs list
        result = '\n'.join(docs)            # Joins all documents into a single string
        # Escaping curly brackets is commented out
        # result = result.replace('{', '{{')
        # result = result.replace('}', '}}')
        return result                       # Returns the cleaned data

# **Main Function**

def main():
    print('main() executed')  # Indicates that the main function has started
    # Sets up parameters for the OpenSearch operations
    index_name1 = 'text_to_sql_index'     # Name of the OpenSearch index to use
    domain = 'https://e0hc00i67ga6mpn1xkxa.us-east-1.aoss.amazonaws.com'  # OpenSearch domain endpoint
    vector_field = 'embeddings_vector'    # Name of the vector field in the index
    fieldname = 'id'                      # Name of the text field in the index
    try:
        # Initializes an instance of the EmbeddingBedrockOpenSearch class
        ebropen = EmbeddingBedrockOpenSearch(domain, vector_field, fieldname)
        # Checks if the index exists in OpenSearch
        ebropen.check_if_index_exists(
            index_name=index_name1,
            region='us-east-1',
            host=domain,
            http_auth=awsauth
        )

        # Retrieves the OpenSearch vector search client
        vcindxdoc = ebropen.getDocumentfromIndex(index_name=index_name1)

        # Defines a user query for the similarity search
        user_query = 'show me all the titles in US region'
        # Performs the similarity search to get relevant documents
        document = ebropen.getSimilaritySearch(user_query, vcindex=vcindxdoc)
        # print(document)  # Commented out; would print the list of documents

        # Processes the retrieved documents to extract data
        # result = ebropen.format_metadata(document)  # Alternative formatting method (commented out)
        result = ebropen.get_data(document)          # Retrieves and cleans the data from documents

        print(result)                                # Prints the final result
    except Exception as e:
        print(e)                                     # Prints any exception that occurs
        traceback.print_exc()                        # Prints the stack trace for debugging

    logger.info(vcindxdoc)                           # Logs information about the vector index document

# Checks if the script is being run directly
if __name__ == '__main__':
    main()  # Calls the main function to execute the script
