from langchain_text_splitters import (RecursiveCharacterTextSplitter, CharacterTextSplitter)

text = """Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language model generation to create more accurate, reliable, and context-aware AI responses. Instead of relying solely on the knowledge stored during model training, RAG systems first search through a knowledge base (like documents, PDFs, or databases) to find relevant information, and then feed that retrieved information to a large language model (LLM) which generates a response based on both the retrieved context and the user's query. This approach is particularly powerful because it solves the "hallucination problem"—where AI models make up plausible-sounding but incorrect information—by grounding responses in actual data. In a typical RAG pipeline, user queries are converted into embeddings (numerical representations), searched against a vector database to retrieve semantically similar documents or chunks, and then those retrieved chunks are passed to the LLM along with the original query so it can generate an informed answer. This makes RAG systems ideal for enterprise applications, customer support, documentation Q&A, and any scenario where accuracy, recency, and source attribution matter more than raw creativity—which is exactly what you're building with your Qdrant, Sentence Transformers, and Groq setup for multi-tenant use cases."""

# there are several chunking strategies used as per requirements. The most common ones are:

# 1.Fixed-size chunking: This strategy involves splitting the text into chunks of a predetermined size, such as 500 or 1000 characters. It is simple to implement but may split sentences or paragraphs inappropriately, potentially losing context.

fixed = CharacterTextSplitter(
    separator="",
    chunk_size=50,
    chunk_overlap=0,
)

print("Fixed-size chunking:")

for i, chunk in enumerate(fixed.split_text(text), 1):
    print(f"Chunk {i}: {chunk}\n")
    print("------------------------------------------------------------")


# 2.Paragraph-based chunking: This strategy splits the text into chunks based on paragraph boundaries. It preserves the natural structure of the text, making it easier to maintain context. However, it may result in uneven chunk sizes, especially if paragraphs vary significantly in length.

paragraph = RecursiveCharacterTextSplitter(
    separators=".",
    chunk_size=10,
    chunk_overlap=0,
)

print("Paragraph-based chunking:")

for i, chunk in enumerate(paragraph.split_text(text), 1):
    print(f"Chunk {i}: {chunk}\n")

# 3.Recursive chunking: This strategy recursively splits the text into smaller chunks based on a hierarchy of separators, such as paragraphs, sentences, and words. It aims to balance context preservation with manageable chunk sizes. However, it can be more complex to implement and may require fine-tuning of parameters.

Recursive = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=120,
    chunk_overlap=20,
)

print("Recursive chunking:")

for i, chunk in enumerate(Recursive.split_text(text), 1):
    print(f"Chunk {i}: {chunk}\n")
