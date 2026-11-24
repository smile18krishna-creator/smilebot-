from datetime import datetime, timedelta, timezone
import os
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Free tier-ல் அதிக availability கொண்ட model
MODEL_ID = "gemini-3.5-flash-lite"


def get_tamil_nadu_now() -> str:
    try:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except ZoneInfoNotFoundError:
        ist_fixed = timezone(timedelta(hours=5, minutes=30), name="IST")
        now = datetime.now(ist_fixed)
    return now.strftime("%A, %B %d, %Y at %I:%M %p IST (Tamil Nadu)")


def get_config(use_search: bool = False) -> types.GenerateContentConfig:
    tn_now = get_tamil_nadu_now()
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None
    return types.GenerateContentConfig(
        system_instruction=(
            f"You are a helpful assistant. "
            f"Current local date and time in Tamil Nadu: {tn_now}."
        ),
        tools=tools,
    )


def create_chat(history=None, use_search: bool = False):
    return client.chats.create(
        model=MODEL_ID,
        config=get_config(use_search=use_search),
        history=history or [],
    )


# Free tier-ல் 429 error தவிர்க்க ஆரம்பத்திலேயே use_search=False வைக்கலாம்
# அல்லது Google AI Studio-வில் Billing/Quota சரிபார்க்கலாம்
search_enabled = False
chat = create_chat(use_search=search_enabled)
turn_count = 0
total_session_tokens = 0
MAX_TURNS = 3

print(f"Chat started ({get_tamil_nadu_now()})! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        print(
            f"\n[Session Ended ({get_tamil_nadu_now()}) | Total Tokens Used: {total_session_tokens}]"
        )
        break

    try:
        response = chat.send_message(user_input)
    except errors.ClientError as e:
        if e.code == 429:
            print(
                "\n[Error 429: Daily/Minute quota exceeded on this API key/model. "
                "Check https://aistudio.google.com/rate-limit or try another API key/project.]\n"
            )
            continue
        raise

    reply_text = response.text or "[Empty response received—daily quota may be exhausted]"
    print(f"\nGemini: {reply_text}")

    usage = response.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = (usage.candidates_token_count or 0) if usage else 0
    turn_tokens = (usage.total_token_count or (input_tokens + output_tokens)) if usage else 0
    total_session_tokens += turn_tokens

    history_token_info = client.models.count_tokens(
        model=MODEL_ID,
        contents=chat.get_history(),
    )

    print(
        f"\n[Tokens -> Input: {input_tokens} | "
        f"Output: {output_tokens} | "
        f"Turn Total: {turn_tokens} | "
        f"Active Memory: {history_token_info.total_tokens}]\n"
    )

    turn_count += 1

    if turn_count % MAX_TURNS == 0:
        recent_history = chat.get_history()[-4:]
        chat = create_chat(history=recent_history, use_search=search_enabled)

        trimmed_info = client.models.count_tokens(
            model=MODEL_ID,
            contents=chat.get_history(),
        )
        print(
            f"[Info: Old chat tokens cleared! "
            f"Active Memory reduced to: {trimmed_info.total_tokens} tokens]\n"
        )

    time.sleep(1)