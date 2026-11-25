import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Loads variables from .env into os.environ
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from environment or .env file.")

# Explicitly pass api_key from os.getenv() (or omit it to let genai.Client() read os.environ["GEMINI_API_KEY"])
client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-3.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
    ),
)

response = chat.send_message("Hello! Can you summarize what we're building?")
print(response.text)