from langchain_aws import BedrockLLM

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock

class LanguageModel():
    def __init__(self,client):
        self.bedrock_client = client
        ############
        # Anthropic Claude     
        # Bedrock LLM
        inference_modifier = {
               ### "max_tokens_to_sample": 3000,
                "temperature": 0,
                "top_k": 20,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"],

            #### "max_tokens_to_sample": 3000, (Commented Out)
            #Purpose: Would limit the maximum number of tokens (words or word pieces) the model can generate to 3000.
            #Note: Commented out with ###, so it's inactive.

            #"temperature": 0,
            #Purpose: Controls the randomness of the output.
            #Explanation: A value of 0 makes the output deterministic (less random).

            #"top_k": 20,
            #Purpose: Limits the number of highest-probability vocabulary tokens to consider.
            #Explanation: The model chooses the next word from the top 20 most probable words.

            #"top_p": 1,
            #Purpose: Sets nucleus sampling parameter.
            #Explanation: Includes all tokens (since 1 means 100% of the probability mass).

            #"stop_sequences": ["\n\nHuman:"],
            #Purpose: Defines sequences at which the model should stop generating text.
            #Explanation: If the model outputs \n\nHuman:, it will stop further text generation.
            }
        print("bedrockllm")
        #self.llm = BedrockLLM(
        #    model_id = "anthropic.claude-v2:1",
            #model_id = "anthropic.claude-3-sonnet-20240229-v1:0",
        #                    client = self.bedrock_client, 
        #                    model_kwargs = inference_modifier 
        #                    )
        
        self.llm = ChatBedrock(
            #model_id = "anthropic.claude-v2:1",
            model_id = "anthropic.claude-3-sonnet-20240229-v1:0",
                            client = self.bedrock_client, 
                            model_kwargs = inference_modifier 
                            )
        
        # Embeddings Modules
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client, 
            model_id="amazon.titan-embed-text-v1"
        )