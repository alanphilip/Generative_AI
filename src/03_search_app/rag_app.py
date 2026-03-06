import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# The Knowledge Base (Our "External Memory")
documents = [
    "Project Apollo 11 was the spaceflight that first landed humans on the Moon.",
    "The internal wifi password for the guest network is 'BlueSky99!'.",
    "To reset the manufacturing robot, hold the red button for 5 seconds, then press start.",
    "The cafeteria serves taco tuesday every week at 12:00 PM."
]

# Retrieval System (From previous step)
def get_embedding(text):
    return genai.embed_content(model="models/text-embedding-004", content=text)['embedding']

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

    # print(highest_score)
    return best_doc

# The Generation Step (The "Augmented" part)
def generate_answer(query):
    # Step A: Retrieve relevant context
    print(f"Searching knowledge base for: '{query}'...")
    relevant_context = find_best_match(query, documents)
    print(f"Found context: \"{relevant_context}\"\n")

    # Construct the Prompt
    prompt = f"""
    You are a helpful assistant. 
    Answer the user's question using ONLY the context provided below.
    If the answer is not in the context, say "I don't know."
    
    Context:
    {relevant_context}
    
    User Question: 
    {query}
    """

    # Generate response using the Chat Model
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)

    return response.text

# Ask a question that requires the private data
answer = generate_answer("what's the weather today?")
print(f"AI Answer:\n{answer}")