'''
Focre AI to retrieve answer from the DB only based on user query.
'''

import numpy as np
from dotenv import load_dotenv
from google import genai

# 1. Setup API
# Load the .env file FIRST so the client can find GEMINI_API_KEY automatically
load_dotenv()
client = genai.Client()

# 2. The Knowledge Base (Our "External Memory")
documents = [
    "Project Apollo 11 was the spaceflight that first landed humans on the Moon.",
    "The internal wifi password for the guest network is 'BlueSky99!'.",
    "To reset the manufacturing robot, hold the red button for 5 seconds, then press start.",
    "The cafeteria serves taco tuesday every week at 12:00 PM."
]

# 3. Retrieval System (Updated to new SDK)
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001", # Using the updated embedding model
        contents=text
    )
    return response.embeddings[0].values

def find_best_match(query, docs):
    """Finds the most relevant document for a query."""
    query_vec = np.array(get_embedding(query))
    best_doc = None
    highest_score = -1

    # Simple linear search (In production, use a Vector DB)
    for doc in docs:
        doc_vec = np.array(get_embedding(doc))
        score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))

        if score > highest_score:
            highest_score = score
            best_doc = doc

    return best_doc

# 4. The Generation Step (Updated to new SDK)
def generate_answer(query):
    # Step A: Retrieve relevant context
    print(f"Searching knowledge base for: '{query}'...")
    relevant_context = find_best_match(query, documents)
    print(f"Found context: \"{relevant_context}\"\n")

    # Step B: Construct the Prompt
    prompt = f"""
    You are a helpful assistant. 
    Answer the user's question using ONLY the context provided below.
    If the answer is not in the context, say "I don't know."
    
    Context:
    {relevant_context}
    
    User Question: 
    {query}
    """

    # Step C: Generate response using the new Client
    response = client.models.generate_content(
        model="gemini-2.5-flash", # A fast, modern generation model
        contents=prompt
    )

    return response.text

# 5. Run it
# answer = generate_answer("How do I reset the robot?")
answer = generate_answer("What is the capital of france?")
print(f"AI Answer:\n{answer}")