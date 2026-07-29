import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is NOT SET or UNAVAILABLE")

client = Groq(api_key=my_api_key)
model= "llama-3.3-70b-versatile"

Job_description = """
About the Role
We are seeking a highly skilled and innovative Machine Learning Engineer to join our Artificial Intelligence team. In this role, you will be responsible for designing, developing, and deploying scalable machine learning models and AI systems that solve complex business problems. You will collaborate closely with data scientists, software engineers, and product managers to bring intelligent, data-driven features into our core products.

Key Responsibilities
Model Development: Design, train, and evaluate machine learning models using traditional ML algorithms, deep learning, Natural Language Processing (NLP), or computer vision techniques.

Pipeline Engineering: Build and maintain scalable, robust ML pipelines for data ingestion, feature engineering, and model deployment.

Optimization: Monitor model performance in production and continuously optimize existing AI models for accuracy, latency, and resource efficiency.

Cross-Functional Collaboration: Work alongside data engineers to ensure data quality and collaborate with software engineering teams to smoothly integrate models into production environments.

Research & Innovation: Stay up-to-date with the latest advancements in AI/ML and proactively prototype new techniques to enhance our technology stack.

Required Qualifications
Education: Bachelors or Masters   degree in Computer Science, Engineering, Mathematics, Statistics, or a highly related technical field.

Programming Skills: Advanced proficiency in Python and familiarity with Object-Oriented languages like C++ or Java.

Frameworks: Hands-on experience with popular machine learning and deep learning frameworks such as TensorFlow, PyTorch, Keras, or Scikit-Learn.

Data Handling: Strong understanding of SQL, relational databases, and experience with data processing tools (e.g., Pandas, NumPy).

Deployment: Experience with containerization tools (Docker, Kubernetes) and deploying models on cloud platforms (AWS, Google Cloud Platform, or Microsoft Azure).

Preferred Qualifications
Practical experience working with Large Language Models (LLMs) and generative AI architectures (e.g., Transformers).

Familiarity with MLOps principles and tools (e.g., MLflow, Kubeflow, Weights & Biases) for model tracking and lifecycle management.

A strong portfolio of public AI projects, active contributions to open-source software, or published research in relevant conferences.

What We Offer
Competitive salary and comprehensive equity packages.

Full health, dental, and vision insurance coverage.

Flexible working hours and remote/hybrid work options.

Dedicated annual budget for continuous learning, technical conferences, and certifications."""

RESUME=input("Please enter the candidate resume in text format Only: ")

def ask_llm(system_prompt, user_prompt):
    sys_msg={
        "role": "system",
        "content": system_prompt
    }
    user_msg={
        "role": "user",
        "content": user_prompt
    }
    messages=[sys_msg, user_msg]
    resposne=client.chat.completions.create(model=model, messages=messages)
    answer=resposne.choices[0].message.content
    return answer

def step1_res_extract(RESUME):
    print("STEP 1")
    system_prompt="""
    You are a professional HR assistant. Extract the skills from the candidates resume provided.
    Only return the skills no other information. Do not invent any skillsby yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt=f"""
    Extract the skills from this resume
    {RESUME}
    """
    return ask_llm(system_prompt, user_prompt)

def step2_JD_extract(JD):
    print("step2")
    system_prompt="""
    You are a professional HR assistant. Extract the skills from the Job description  provided.
    Only return the skills no other information. Do not invent any skills by yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt=f"""
    Extract the skills from this JD
    {JD}
    """
    return ask_llm(system_prompt, user_prompt)

def step3_match(candidate,jd):
    print("step3")
    system_prompt="""
    You are a professional HR assistant. compare the skills of candidate and the skills required in the JD and produce a final score between
    1 and 100. also produce a short verdict whther the candidate is a good fit for the role.
    """
    user_prompt=f"""
    Compare and match the skills
    JD:
    {Job_description}
    Candidate:
    {candidate}
    """
    return ask_llm(system_prompt, user_prompt)

candidate=step1_res_extract(RESUME)
print(candidate)
sleep(2)
jd=step2_JD_extract(Job_description)
print(jd)
sleep(2)
score=step3_match(candidate,jd)
print(score)