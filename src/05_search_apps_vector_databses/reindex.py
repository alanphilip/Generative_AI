import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Setup
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

MODEL_NAME = "models/text-embedding-004"
INPUT_FILE = "embedding_index_3m.json"
OUTPUT_FILE = "embedding_index_gemini.json"
BATCH_SIZE = 20  # Number of items to embed in one API call

def create_text_for_embedding(row):
    """Combines title and summary for a richer search context."""
    return f"Title: {row['title']}\nSummary: {row['summary']}"

def reindex_data():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE)

    # Optional: Drop the old OpenAI vector column to save space
    if 'ada_v2' in df.columns:
        df = df.drop(columns=['ada_v2'])

    # 2. Prepare text to be embedded
    # We create a temporary list of strings to send to the API
    texts_to_embed = df.apply(create_text_for_embedding, axis=1).tolist()

    new_embeddings = []

    print(f"Starting reindexing of {len(df)} records...")

    # 3. Batch Processing
    # We loop through the data in chunks (batches) to respect API limits and improve speed
    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE)):
        batch = texts_to_embed[i : i + BATCH_SIZE]

        try:
            # Call Gemini API
            response = genai.embed_content(
                model=MODEL_NAME,
                content=batch,
                task_type="retrieval_document",
                title="Video Entry" # Optional: Descriptive title for the document type
            )

            # Append the resulting vectors to our list
            # Note: The response structure depends on input.
            # If batch, 'embedding' key usually contains a list of vectors.
            batch_embeddings = response['embedding']
            new_embeddings.extend(batch_embeddings)

            # Rate Limit Protection (optional, adjust based on your quota)
            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            # In a production script, you might want to retry or fill with None
            # For now, we append empty lists to keep alignment, or break
            new_embeddings.extend([[]] * len(batch))

    # 4. Assign new vectors to DataFrame
    # Using a new column name to distinguish from old OpenAI vectors
    df['gemini_vector'] = new_embeddings

    # Filter out any failed rows (empty vectors)
    df = df[df['gemini_vector'].apply(len) > 0]

    # 5. Save
    print(f"Saving reindexed data to {OUTPUT_FILE}...")
    df.to_json(OUTPUT_FILE, orient='records')
    print("Done!")

if __name__ == "__main__":
    reindex_data()