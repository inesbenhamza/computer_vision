import os 
from dotenv import load_dotenv

load_dotenv() 

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key: 
    raise ValueError("no openAI key found, please set the OPENAI_API_KEY environment variable")

