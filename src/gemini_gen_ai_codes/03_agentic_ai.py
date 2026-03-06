'''
Intelligently switch between DB and LLM based on user query context.
'''

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types # We need this to configure tools

# 1. Setup API
load_dotenv()
client = genai.Client()

# 2. The Internal Knowledge Base
documents = [
    "Project Apollo 11 was the spaceflight that first landed humans on the Moon.",
    "The internal wifi password for the guest network is 'BlueSky99!'.",
    "To reset the manufacturing robot, hold the red button for 5 seconds, then press start."
]

def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def find_best_match(query, docs):
    query_vec = np.array(get_embedding(query))
    best_doc = None
    highest_score = -1
    for doc in docs:
        doc_vec = np.array(get_embedding(doc))
        score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        if score > highest_score:
            highest_score = score
            best_doc = doc
    return best_doc

# 3. Define the Tools (The Agent's Actions)
# The Docstrings are CRITICAL here. The AI reads them to know how the tool works.

def search_internal_database(query: str) -> str:
    """Use this tool to search the private internal database for company secrets, wifi passwords, and robotics manuals."""
    print(f"\n  [System] Agent is executing: search_internal_database('{query}')")
    return find_best_match(query, documents)

def search_public_web(query: str) -> str:
    """Use this tool to search the public internet for general knowledge, weather, and world facts."""
    print(f"\n  [System] Agent is executing: search_public_web('{query}')")
    # In a real app, you would call a web search API here.
    return "The capital of France is Paris, and the current temperature is 18°C."

# 4. Initialize the Agentic Chat
# We pass the functions directly into the 'tools' array.
agent_chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        tools=[search_internal_database, search_public_web],
        temperature=0.0 # Low temperature makes the agent more analytical and deterministic
    )
)

# 5. Test the Agent's Decision Making
def ask_agent(prompt):
    print(f"\nUser: {prompt}")
    # The SDK handles the loop: it asks the AI, executes the chosen function,
    # sends the result back to the AI, and returns the final text to us.
    response = agent_chat.send_message(prompt)
    print(f"Agent: {response.text}")
    print("-" * 50)

# Query 1: Requires internal data
ask_agent("I have a guest in the lobby, how do they connect to the internet?")

# Query 2: Requires external knowledge
ask_agent("What is the capital of France?")