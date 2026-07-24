import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY is not set in the env.")

client = Groq(api_key=my_api_key)
model = "llama-3.3-70b-versatile"
role = "user"

prompt1 = "hey"
prompt2 = "explain RAG in details"
prompt3 = "tell emerging technologies in AI"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    system_prompt = {
        "role": "system",
        "content": "prompt"
    }

    message = {
        "role": role,
        "content": prompt
    }

    messages = [message]
    response = client.chat.completions.create(model=model, messages=messages)
    usage = response.usage
    
    print(f"Prompt: {prompt}")
    print(f"Tokens used: {usage.total_tokens}") 