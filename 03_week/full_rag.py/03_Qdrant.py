import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

print("Qdrant client initialized successfully.")

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384  # Size of the embeddings from the model

# Delete collection if it already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)


# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

# PART 4 — LOAD OUR KNOWLEDGE

with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]
# ["line 1 ", "line2", "line3"....]
print(f"Loaded {len(documents)} documents")

# PART 5 — CREATE EMBEDDINGS

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2") #384

print("Embedding model ready!")


embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# PART 6 — CREATE QDRANT POINTS

points = []

for i, embedding in enumerate(embeddings):

    point = PointStruct(
        id=i + 1, #id=1

        vector=embedding.tolist(),

        payload={
            "text": documents[i]
        }
    )

    points.append(point)

# PART 7 — UPLOAD TO QDRANT

client.upsert( #upload+insert
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant!")

# PART 8 — SEARCH QDRANT

def search(query, top_k=3):

    # Convert the question into an embedding
    query_vector = model.encode(query).tolist()

    # Search Qdrant for similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results

# PART 9 — TEST SEARCH

query = "How many vacation days do I get?"

results = search(query, top_k=3)

print("\nSearch results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    # print(result.payload["text"])
    print()

# PART 10 — CONNECT TO GROQ

groq_client = Groq(
    api_key=GROQ_API_KEY
)

# PART 11 — ASK THE LLM

def ask_llm(question, context):

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# PART 12 — COMPLETE RAG PIPELINE

question = input("Enter your question: ")

results = search(question, top_k=3)


# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)