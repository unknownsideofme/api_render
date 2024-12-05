from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from pprint import pprint
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import pickle
from langchain.schema import Document





def calc_suggestions(res, title, llm):
    # Convert JSON string into a Document object
    if isinstance(res, str):
        res = json.loads(res)

    # Convert `res` into LangChain-compatible Document objects
    res_document = [
        Document(page_content=json.dumps(value), metadata={"type": key})
        for key, value in res["message"].items()
    ]
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a title verification assistant for the Press Registrar General of India. 
        1. You will be given a response that stores the phonetic similarity, semantic similarity, and string similarity between the input title and existing titles.
        2. You will also be given the prefix and suffix score of the title, indicating how frequently the suffix and prefix of the title are repeated as a percentage of the existing titles.
        3. Your task is to give suggestions to the user based on the phonetic similarity, semantic similarity, string similarity, and prefix and suffix scores so that the title achieves a better acceptance score.
        4. Suggestions can include removing or replacing commonly used prefixes and suffixes or making the title more phonetically and semantically unique.

        Output Format:
        {{
            "suggestions": {{}}
        }}
        
        Input: {input}
        Context: {context}
        """
    )
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Invoke the document chain
    response = document_chain.invoke({"input": title, "context": res_document})
    
    # Parse the response and return suggestions
    return response
