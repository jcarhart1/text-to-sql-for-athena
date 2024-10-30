import boto3                       # Imports the AWS SDK for Python, which allows interaction with AWS services
from botocore.config import Config # Imports the Config class for setting configurations like retries and timeouts
import logging                     # Enables logging of messages for debugging and tracking program execution

# Sets up a logger to record debug-level messages and outputs them to the console
logger = logging.getLogger(__name__)   # Creates a logger with the name of the current module
logger.setLevel(logging.DEBUG)         # Sets the logger to capture all levels of messages at and above DEBUG
logger.addHandler(logging.StreamHandler())  # Adds a handler to output log messages to the console

# Defines a retry configuration for AWS clients to handle transient errors and throttling
retry_config = Config(
    region_name='us-east-1',  # Specifies the AWS region where the services are located
    retries={
        'max_attempts': 10,   # Sets the maximum number of retry attempts for failed requests
        'mode': 'standard'    # Uses the standard retry mode provided by AWS SDK
    }
)

# Defines a class to encapsulate methods for creating AWS service clients
class Clientmodules():
    def __init__(self):
        pass  # The constructor does nothing in this case

    # Creates a client for AWS Bedrock service
    def createBedrockClient():
        session = boto3.session.Session()  # Starts a new session with AWS
        bedrock_client = session.client('bedrock', config=retry_config)  # Creates a Bedrock client with retry config
        logger.info('bedrock client created for profile')  # Logs the creation of the Bedrock client
        return bedrock_client  # Returns the Bedrock client instance

    # Creates a client for AWS Bedrock Runtime service
    def createBedrockRuntimeClient():
        session = boto3.session.Session()  # Starts a new session with AWS
        bedrock_runtime_client = session.client('bedrock-runtime', config=retry_config)  # Creates a Bedrock Runtime client
        logger.info('bedrock runtime client created')  # Logs the creation of the Bedrock Runtime client
        return bedrock_runtime_client  # Returns the Bedrock Runtime client instance

    # Creates a client for Amazon Athena service
    def createAthenaClient():
        session = boto3.session.Session()  # Starts a new session with AWS
        athena_client = session.client('athena', config=retry_config)  # Creates an Athena client
        logger.info('athena client created')  # Logs the creation of the Athena client
        return athena_client  # Returns the Athena client instance

    # Creates a client for Amazon S3 service
    def createS3Client():
        session = boto3.session.Session()  # Starts a new session with AWS
        s3_client = session.client('s3', config=retry_config)  # Creates an S3 client
        logger.info('s3 client created !!')  # Logs the creation of the S3 client
        return s3_client  # Returns the S3 client instance
