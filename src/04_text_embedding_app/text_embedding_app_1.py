import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# Returns Vector
def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result['embedding']

# Define the Cosine Similarity function
def cosine_similarity(v1, v2):
    # Convert lists to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Calculate dot product and magnitudes (norms)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    return dot_product / (magnitude_v1 * magnitude_v2)

# The Data: 3 sentences
text_1 = "I love writing Python code."
text_2 = "Programming is my favorite hobby."  # Similar to 1
text_3 = "This pizza tastes delicious."       # Different from 1 & 2

# Generate Embeddings
vec_1 = get_embedding(text_1)
vec_2 = get_embedding(text_2)
vec_3 = get_embedding(text_3)

# Compare them
similarity_1_2 = cosine_similarity(vec_1, vec_2) # Coding vs Programming
similarity_1_3 = cosine_similarity(vec_1, vec_3) # Coding vs Pizza

print(f"Similarity (Coding vs Programming): {similarity_1_2:.4f}")
print(f"Similarity (Coding vs Pizza):       {similarity_1_3:.4f}")