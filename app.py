import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env into the environment
load_dotenv()

# The client automatically reads GEMINI_API_KEY from the environment
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Explain virtual environments in one sentence.",
)

print(response.text)