'''
Return the sentence that closely match with the user query which is stored in the document DB.
Semantic Search uses does context match using cosine similarity.
'''

import os
from dotenv import load_dotenv
from google import genai
import numpy as np

# 1. Load the environment variables FIRST
load_dotenv()

# 2. Custom error handling (Optional but good practice)
if "GEMINI_API_KEY" not in os.environ:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# 3. Initialize the client
client = genai.Client()

# 4. Define the model
MODEL_NAME = "gemini-embedding-001"

# 2. Our "Database" of documents
documents = [
    "The best way to cook a steak is to sear it on high heat.",
    "Machine learning models require large datasets for training.",
    "The Golden State Warriors won the NBA championship.",
    "Generative AI can create images and text from prompts.",
    "To fix a flat tire, you need a jack and a lug wrench."
]

# 3. Helper function for embeddings
def get_embedding(text):
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=text
    )
    # The response is now a Pydantic object, not a dictionary.
    return response.embeddings[0].values

# 4. "Indexing" Step
print("Indexing documents...")
doc_embeddings = []
for word in documents:
    doc_embeddings.append({
        "text": word,
        "vector": np.array(get_embedding(word))
    })
print("Indexing complete!\n")

# 5. The Search Function
def search_engine(query):
    query_vector = np.array(get_embedding(query))

    results = []

    # Compare query against every document
    for data in doc_embeddings:
        doc_vector = data["vector"]

        # Calculate Cosine Similarity manually
        dot_product = np.dot(query_vector, doc_vector)
        norm_a = np.linalg.norm(query_vector)
        norm_b = np.linalg.norm(doc_vector)
        similarity = dot_product / (norm_a * norm_b)

        results.append((similarity, data["text"]))

    # Sort results by score (highest first)
    results.sort(key=lambda x: x[0], reverse=True)

    # Print the Top Match
    print(f"Query: '{query}'")
    print(f"Top Match: '{results[0][1]}'")
    print(f"Confidence Score: {results[0][0]:.4f}")
    print("-" * 30)

# 6. Test it out!
search_engine("How do I prepare beef?")
search_engine("Tell me about artificial intelligence.")
search_engine("sports news")