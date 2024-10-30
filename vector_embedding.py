
# This program is designed to process a set of documents (specifically, an IMDb schema in JSON Lines format) and
# convert them into numerical representations called embeddings using Amazon's Bedrock service. These embeddings
# capture the semantic meaning of the documents, allowing for efficient similarity searches and other natural language
# processing tasks. The script utilizes the FAISS (Facebook AI Similarity Search) library to store these embeddings i
# n a vector store, which can be saved locally for later use. The main functionalities include creating embeddings
# from documents, saving the vector store locally, loading the vector store when needed, and formatting
# metadata from the documents.


##from langchain.document_loaders import JSONLoader
from langchain_community.document_loaders import \
    JSONLoader  # Imports the JSONLoader to load JSON documents for processing
import logging  # Enables logging of events and errors for debugging purposes
import json  # Provides functions for working with JSON data formats
import os  # Offers a way to interact with the operating system (e.g., file paths)
import sys  # Provides access to system-specific parameters and functions
import re  # Supports regular expression matching operations for advanced string processing
# sys.path.append("/home/ec2-user/SageMaker/llm_bedrock_v0/")  # (Commented out) Would add a specific directory to the system path for module access
import warnings  # Allows control over warning messages that the program might produce

# This line suppresses specific warning messages from the 'pydantic' module to keep the console output clean
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")

# import schema_details.tbl_schema as sc                     # (Commented out) Possibly intended for schema details, not used here
from llm_basemodel import LanguageModel  # Imports a custom LanguageModel class for language model operations
from boto_client import Clientmodules  # Imports a custom module to create AWS service clients

# from langchain.embeddings import BedrockEmbeddings
from langchain_community.embeddings import \
    BedrockEmbeddings  # Imports the BedrockEmbeddings class for generating embeddings
from langchain_aws import \
    BedrockEmbeddings  # Also imports BedrockEmbeddings, possibly redundant but ensures compatibility

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import \
    FAISS  # Imports the FAISS vector store for efficient similarity search operations

import numpy as np  # Imports NumPy for numerical operations, often used for handling arrays and matrices
from datetime import datetime  # Provides classes for manipulating dates and times


# **Class Definition for Embedding Operations**
class EmbeddingBedrock:
    # The constructor method initializes the class and sets up necessary attributes
    def __init__(self):
        # Creates a Bedrock runtime client using a custom AWS client module
        self.bedrock_client = Clientmodules.createBedrockRuntimeClient()

        # Initializes the LanguageModel with the Bedrock client to access language model functionalities
        self.language_model = LanguageModel(self.bedrock_client)

        # Stores the language model instance for generating text or embeddings
        self.llm = self.language_model.llm

        # Stores the embeddings model instance for converting text into numerical vectors
        self.embeddings = self.language_model.embeddings

        # Sets the identifier for the embeddings model used, helpful for version tracking
        self.embeddings_model_id = 'amazon.titan-embed-text-v2:0'  # Specifies the updated embedding model version

    # Method to create embeddings from a set of documents
    def create_embeddings(self):
        # Loads documents from a JSON Lines file using JSONLoader
        documents = JSONLoader(
            file_path='imdb_schema.jsonl',  # Path to the JSON Lines file containing the documents (IMDb schema)
            jq_schema='.',  # JSON query to select the entire content of each JSON object
            text_content=False,  # Indicates that the content is not plain text but structured JSON
            json_lines=False  # Indicates that the file is in JSON Lines format (one JSON object per line)
        ).load()  # Executes the loading process and returns a list of documents

        # Reaffirms the embeddings model ID to ensure consistency
        embeddings_model_id = 'amazon.titan-embed-text-v2:0'  # Ensures the correct embeddings model is used

        try:
            # Creates a vector store from the documents using the embeddings model
            vector_store = FAISS.from_documents(documents, self.embeddings)
        except Exception:
            # Raises an exception if the vector store creation fails
            raise Exception("Failed to create vector store")

        # Prints a confirmation message indicating successful creation
        print("Created vector store")

        # Returns the created vector store for further use
        return vector_store

    # Method to save the vector store to a local directory
    def save_local_vector_store(self, vector_store, vector_store_path):
        # Captures the current date and time in a specific format for unique naming
        time_now = datetime.now().strftime("%d%m%Y%H%M%S")

        # Constructs the full path for saving the vector store, appending a timestamp and file extension
        vector_store_path = vector_store_path + '/' + time_now + '.vs'

        # Retrieves the embeddings model ID used for creating the vector store
        embeddings_model_id = self.embeddings_model_id

        try:
            # Checks if the vector_store_path is empty and sets a default path if necessary
            if vector_store_path == "":
                vector_store_path = f"../vector_store/{time_now}.vs"

            # Ensures that the directory structure exists; creates it if it doesn't
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)

            # Saves the vector store to the specified local path
            vector_store.save_local(vector_store_path)

            # Writes the embeddings model ID to a file within the vector store directory for future reference
            with open(f"{vector_store_path}/embeddings_model_id", 'w') as f:
                f.write(embeddings_model_id)
        except Exception:
            # Prints an error message but continues execution even if saving fails
            print("Failed to save vector store, continuing without saving...")

        # Returns the path where the vector store was saved
        return vector_store_path

    # Method to load a vector store from a local directory
    def load_local_vector_store(self, vector_store_path):
        try:
            # Reads the embeddings model ID from the saved file to ensure compatibility
            with open(f"{vector_store_path}/embeddings_model_id", 'r') as f:
                embeddings_model_id = f.read()

            # Loads the vector store from the specified path using the embeddings model
            vector_store = FAISS.load_local(vector_store_path, self.embeddings)

            # Prints a confirmation message indicating successful loading
            print("Loaded vector store")

            # Returns the loaded vector store for use in the program
            return vector_store
        except Exception:
            # Prints an error message if loading fails and indicates that a new vector store will be created
            print("Failed to load vector store, continuing creating one...")

    # Method to format metadata extracted from the documents for display or further processing
    def format_metadata(self, metadata):
        docs = []
        # Iterates over each element in the metadata list
        for elt in metadata:
            # Retrieves the content of the current document
            processed = elt.page_content

            # Removes indentation and line feed characters to clean up the text
            for i in range(20, -1, -1):
                processed = processed.replace('\n' + ' ' * i, '')

            # Appends the cleaned text to the docs list
            docs.append(processed)

        # Joins all the documents into a single string, separated by newline characters
        result = '\n'.join(docs)

        # Escapes curly brackets to prevent issues in formats that interpret them specially
        result = result.replace('{', '{{')
        result = result.replace('}', '}}')

        # Returns the formatted metadata string
        return result


# **Main Function to Execute Embedding Creation and Save the Vector Store**
def main():
    # Creates an instance of the EmbeddingBedrock class to access its methods
    embedding_bedrock = EmbeddingBedrock()

    # Calls the create_embeddings method to generate the vector store from documents
    vector_store = embedding_bedrock.create_embeddings()

    # Specifies the path where the vector store will be saved locally
    vector_store_path = './vector_store'

    # Calls the method to save the vector store at the specified path
    embedding_bedrock.save_local_vector_store(vector_store, vector_store_path)


# Checks if this script is being run directly (not imported as a module)
if __name__ == '__main__':
    main()  # Executes the main function to run the program
