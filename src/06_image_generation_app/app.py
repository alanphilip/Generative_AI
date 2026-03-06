import os
from PIL import Image
from google import genai
from google.genai import types
from io import BytesIO
import dotenv

# Load environment variables (Make sure you have GEMINI_API_KEY in your .env)
dotenv.load_dotenv()

# Initialize the Gemini Client
try:
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# Path Setup
image_dir = os.path.join(os.curdir, 'images')
image_path = os.path.join(image_dir, 'generated-image.png')
output_path = os.path.join(image_dir, 'generated_variation.png')

# Load the existing image to show it
print(f"Opening original: {image_path}")
img = Image.open(image_path)
img.show()

try:
    print("LOG: Requesting image variation from Gemini...")

    # Open image as bytes to send to the API
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Gemini 2.0/2.5 Flash models support native image output
    # We use a prompt to describe the 'variation' we want.
    response = client.models.generate_content(
        model="imagen-4.0-ultra-generate",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            "Create a variation of this image. Keep the same style and subject, but change the lighting and composition slightly."
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"] # This tells Gemini to respond with a new image
        )
    )

    # 3. Process the response
    print("LOG: Processing generated image...")
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            # Convert the returned bytes back into an image
            generated_img = Image.open(BytesIO(part.inline_data.data))
            generated_img.save(output_path)
            print(f"Success! Saved to {output_path}")
            generated_img.show()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Completed!")