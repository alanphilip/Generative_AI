import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Explicitly pass the API key to ensure it's not trying to use Vertex AI
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print(f"{'Model Name':<30} | {'Actions'}")
print("-" * 50)

try:
    # Get all models
    models = client.models.list()

    for m in client.models.list():
        print(m.name)

except Exception as e:
    print(f"An error occurred: {e}")

