import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Setup
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# Select the embedding model
# "models/text-embedding-004" is a common choice, but "models/embedding-001" is also available
MODEL_NAME = "models/text-embedding-004"

SIMILARITIES_RESULTS_THRESHOLD = 0.55
DATASET_NAME = "embedding_index_gemini.json"

def load_dataset(source: str) -> pd.DataFrame:
    """Load the video session index"""
    # Assuming the JSON structure matches what you had before.
    # Note: If your dataset 'ada_v2' column contains OpenAI embeddings (1536 dimensions),
    # they will not work with Gemini embeddings (768 dimensions for text-embedding-004).
    # You would need to re-generate embeddings for your dataset using Gemini first.
    pd_vectors = pd.read_json(source)
    return pd_vectors.drop(columns=["text"], errors="ignore").fillna("")

def cosine_similarity(a, b):
    # Handle potential dimension mismatch if comparing different model outputs
    if len(a) > len(b):
        b = np.pad(b, (0, len(a) - len(b)), 'constant')
    elif len(b) > len(a):
        a = np.pad(a, (0, len(b) - len(a)), 'constant')

    # Calculate similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def get_videos(query: str, dataset: pd.DataFrame, rows: int) -> pd.DataFrame:
    # Create a copy of the dataset
    video_vectors = dataset.copy()

    # Get the embeddings for the query using Gemini
    result = genai.embed_content(
        model=MODEL_NAME,
        content=query,
        task_type="retrieval_query" # Optimizes embedding for search queries
    )
    query_embeddings = result['embedding']

    # Create a new column with the calculated similarity for each row
    # Note: Ensure the vector column name matches your dataset (e.g., 'gemini_vector')
    video_vectors["similarity"] = video_vectors["gemini_vector"].apply(
        lambda x: cosine_similarity(np.array(query_embeddings), np.array(x))
    )

    # Filter the videos by similarity
    mask = video_vectors["similarity"] >= SIMILARITIES_RESULTS_THRESHOLD
    video_vectors = video_vectors[mask].copy()

    # Sort the videos by similarity
    video_vectors = video_vectors.sort_values(by="similarity", ascending=False).head(rows)

    return video_vectors.head(rows)

def display_results(videos: pd.DataFrame, query: str):
    def _gen_yt_url(video_id: str, seconds: int) -> str:
        return f"https://youtu.be/{video_id}?t={seconds}"

    print(f"\nVideos similar to '{query}':")

    if videos.empty:
        print("No videos found above the similarity threshold.")
        return

    for _, row in videos.iterrows():
        youtube_url = _gen_yt_url(row["videoId"], int(row["seconds"]))
        # Handle cases where summary might be short or empty
        summary_text = str(row['summary'])
        summary_preview = ' '.join(summary_text.split()[:15])

        print(f" - {row['title']}")
        print(f"   Summary: {summary_preview}...")
        print(f"   YouTube: {youtube_url}")
        print(f"   Similarity: {row['similarity']:.4f}")
        print(f"   Speakers: {row['speaker']}")

# Main Execution
if __name__ == "__main__":
    # Ensure dataset exists before loading
    if not os.path.exists(DATASET_NAME):
        print(f"Error: Dataset file '{DATASET_NAME}' not found.")
    else:
        pd_vectors = load_dataset(DATASET_NAME)

        # Get user query from input
        while True:
            query = input("\nEnter a query (or type 'exit'): ")
            if query.lower() == "exit":
                break

            try:
                videos = get_videos(query, pd_vectors, 5)
                display_results(videos, query)
            except Exception as e:
                print(f"An error occurred: {e}")