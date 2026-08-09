import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is NOT SET or UNAVAILABLE")

client = Groq(api_key=my_api_key)

model= "llama-3.3-70b-versatile"
prompt = """Explain how streaming works in llm models"""
messages = {
    "role": "user",
    "content": prompt
}
messages = [messages]

stream = client.chat.completions.create(model=model, messages=messages, stream=True)
for chunks in stream:
    content = chunks.choices[0].delta.content
    if content:
        print(content, end="", flush=True)