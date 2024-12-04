from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from pprint import pprint
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API Key not set. Please set the OPENAI_API_KEY environment variable.")

langsmith_api_key = os.getenv("langsmith_api_key")




# Set additional environment variables programmatically
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "SLIFTEX"
# Paths for FAISS index and metadata
faiss_index_path = "faiss_index"
metadata_path = "faiss_metadata.pkl"

# Load FAISS index with OpenAI Embeddings
db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Load metadata from pickle
with open(metadata_path, "rb") as file:
    db.docstore = pickle.load(file)
    
    
context = db


# Initialize the LLM model (OpenAI GPT-3.5-turbo)
llm = ChatOpenAI(model = "gpt-3.5-turbo", api_key=api_key)

def Pref_suff (context ,llm , db  ):
    
    prompt = ChatPromptTemplate.from_template("""
    You are a title verification assistant for the Press Registrar General of India.

    1. You are given a list of existing titles.
    2. Your task is to find out the list of most used prefixes and suffixes of the title.
    3. Return the list of prefixes and suffixes with a percentage that represents the frequency of the prefix or suffix in the titles.
    4. % = (number of titles with the prefix or suffix / total number of titles) * 100.
    5. Only find out the list of top 40 prefixes and suffixes.

    Output Format:
    {{
        "prefixes": {{
            "prefix1": 10,
            "prefix2": 5,
            ...
        }},
        "suffixes": {{
            "suffix1": 15,
            "suffix2": 12,
            ...
        }}
    }}

    context : {context}
    """

    )
    retriever = db.as_retriever()
    document_chain_semantic = create_stuff_documents_chain(llm, prompt)
    retreival_chain_semantic = create_retrieval_chain(retriever, document_chain_semantic)
    
    res = retreival_chain_semantic.invoke({ "input": context})
    return res['answer']


print( Pref_suff (context ,llm , db  ))