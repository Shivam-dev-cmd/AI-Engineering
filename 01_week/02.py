import os 
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY is not set in the env.")

client = Groq(api_key=my_api_key)
model= "llama-3.3-70b-versatile"
role = "user"
temperature = 0.7

system_prompt = {
    
    "role": "system",
    "content": "you are an strict college proffessor and you will answer the questions in a very strict manner."
}

prompt = "how to cover syllabus in 2 months?" 

message = {
    "role": role,
    "content": prompt
}

messages = [system_prompt, message]
response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
print(response.choices[0].message.content)