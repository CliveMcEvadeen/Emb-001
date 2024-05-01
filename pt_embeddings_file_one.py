from rich.console import Console
import os
from dotenv import load_dotenv
from langchain_community.embeddings.bedrock import BedrockEmbeddings

# openAI embeddings.
from langchain_openai import OpenAIEmbeddings

# console object
con = Console()

load_dotenv()

def Get_openai_embedding(model):
    api_key_ = os.getenv("open_ai_key")
    embeddings = OpenAIEmbeddings(model=model, api_key=api_key_)
    return embeddings

def get_embeddings_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )

    return embeddings
