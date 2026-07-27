import os 
from pathlib import Path 
from time import sleep
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is NOT SET or UNAVAILABLE")

client = Groq(api_key=my_api_key)
model = "llama-3.3-70b-versatile"

def get_product_price(product):
    if product == "Dominar 400":
        return 250000
    elif product == "Dominar 250":
        return 190000
    else:
        return 0

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Arithmatic Error !!"

tools ={
    "get_product_price": get_product_price,
    "calculator": calculator
}

system_prompt = """
you are an slaes executive aslo an shopping assistant.

you have these tools:

get_product_price(product),
calculator(expression)
IMPORTANT: call thses tools exactly like these examples:

Action: get_product_price("Dominar 400")
Action: calculator("300000 - 250000")

Never write:
get_product_price(product="Dominar400")

Never write:
calculator(expression="300000 - 250000")

Follow these rules:

1.Decide what you need to do next.
2.call one tool at a time.
3.After writing an Action, STOP immediately.
4.Never guessor invent a tool result.
5.Wait untill you recive an obeservation.
6.Then decide your next action.
7.When the task is complete, give the final answer.

Format:

Thought: what you need to do 
Action: tool_name(arguemnt)

when finished:

Final answer: your answer
"""

def run_agent(question):
    messages =[
        {
            "role":"system",
            "content": system_prompt
        },
        {
            "role":"user",
            "content": question
        }
    ]

    for step in range(5):

        print("\n----------------")
        print("STEP",step + 1)
        print("----------------")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
            
        )

        answer = response.choices[0].message.content

        print(answer)

        if "Final answer:" in answer:
            break

        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

        if tool_name in tools:
            tool = tools[tool_name]
            observation = tool(tool_input)

        else:
            observation = "tool not found"

        print(
            "observation:",
            observation
        )

        messages.append({
            "role":"assistant",
            "content": answer
        }) 

        messages.append({
            "role":"assistant",
            "content": answer
        })

        messages.append({
            "role":"user",
            "content":
                "observation:" + str(observation)
        
        })
        sleep(5)

prompt = """
i have budget of 300000 rupees.now tell me whats the price of an dominar 400? and if i have to buy the dominar 400 how much do i need to spend and how much i will be left with.
"""
run_agent(prompt)