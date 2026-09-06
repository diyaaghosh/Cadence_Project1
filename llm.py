import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    grok_api_key = os.getenv("GROQ_API_KEY" )
    if not grok_api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    llm = ChatGroq(api_key=grok_api_key,model_name="qwen/qwen3.6-27b",temperature=0.01, max_tokens=1024)
    return llm
