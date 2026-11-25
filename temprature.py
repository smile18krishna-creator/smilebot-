import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
stream = client.models.generate_content_stream(
    model="gemini-3.5-fash",  
    contents="Explain what Python is in 3 short sentences.",
    config=types.GenerateContentConfig(temperature=0.7),
)

for part in stream:
    if part.text:
        print(part.text, end="", flush=True)

print()