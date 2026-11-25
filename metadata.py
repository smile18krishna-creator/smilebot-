import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

start_time = time.perf_counter()  # Start time
first_token_time = None
last_chunk = None

stream = client.models.generate_content_stream(
    model="gemini-3.5-flash",
    contents="Explain metadata in two short sentences.",
    config=types.GenerateContentConfig(temperature=0.3),
)

# 1. Stream text & capture First Token Time
print("Answer: ", end="", flush=True)
for chunk in stream:
    if chunk.text:
        if first_token_time is None:
            first_token_time = time.perf_counter()  # First token time
        print(chunk.text, end="", flush=True)
    last_chunk = chunk

end_time = time.perf_counter()  # End time
print("\n")

# Time calculations
ttft = (first_token_time - start_time) if first_token_time is not None else None
total_time = end_time - start_time

print("--- Timing Metadata ---")
print(f"Start time (perf_counter): {start_time:.4f}s")
if ttft is not None:
    print(f"First token time (TTFT):   {ttft:.4f}s")
print(f"End time (perf_counter):   {end_time:.4f}s")
print(f"Total response time:       {total_time:.4f}s")

# 2. Token usage metadata (from the final chunk)
usage = last_chunk.usage_metadata if last_chunk else None
print("\n--- Token Usage Metadata ---")
if usage:
    print("Prompt tokens:", usage.prompt_token_count)
    print("Response tokens:", usage.candidates_token_count)
    print("Total tokens:", usage.total_token_count)
else:
    print("Usage metadata not available.")

# 3. Finish reason & grounding metadata
print("\n--- Candidate Metadata ---")
if last_chunk and last_chunk.candidates:
    candidate = last_chunk.candidates[0]
    print("Finish reason:", candidate.finish_reason)
    print("Grounding/Citation metadata:", candidate.grounding_metadata)