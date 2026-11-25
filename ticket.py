import json
import os
import time
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

client = genai.Client(api_key=API_KEY)

SAMPLE_TICKETS = [
    "yo wtf?? i got charged $249.99 TWICE on my visa ending 4821 this morning!! i only signed up for the starter plan not enterprise. fix this or im calling my bank for a chargeback asap",
    "Hi team, I just got 4 password reset emails I didn't request and now my 2FA phone number was changed to a number in another country?? Please lock my account immediately I think someone hacked me.",
    "app keeps crashing every single time i click export to pdf!! error code 502 pops up and i have a client presentation in 20 mins pls help",
    "Loving the new dashboard update btw!! Quick idea: would be super cool if we could schedule weekly CSV reports to go straight to Slack every Monday morning. No rush, just thought I'd share :)",
    "Oh great, another $15 'mystery add-on fee' on my invoice that I definitely never clicked. Love paying extra for stuff I don't use. Can someone take this off next month's bill?",
    "Your sync bug wiped three weeks of our team's project notes yesterday. Our legal counsel is reviewing your SLA terms right now and we will pursue damages if data isn't restored by 5 PM.",
    "hello i changed my work email after our company rebranded from oldco.com to newco.io and now SSO wont let me sign in. can u update my login email?",
    "hey so dark mode looks kinda weird on safari like the sidebar text is grey on grey barely readable also while ur fixing css pls add keyboard shortcuts for archiving tasks thx",
    "Hey! Just renewed our annual team workspace—everything has been running smoothly. Could you send over a GST/tax invoice with our company address for our accounting folder?",
    "cant log in on my new phone bcuz my old phone screen broke and google authenticator was on it. still know my password tho how do i reset 2fa",
]


class TicketAnalysis(BaseModel):
    category: Literal["billing", "technical_bug", "account_access", "feature_request"] = Field(
        description="Primary issue category. If mixed with a bug, prioritize the broken functionality."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high = active outage, immediate deadline, security breach, or >$100 dispute; medium = blocked login or minor billing issue; low = cosmetic bug, invoice copy, or feature idea."
    )
    sentiment: Literal["positive", "neutral", "angry"] = Field(
        description="Customer emotional tone, accounting for sarcasm."
    )
    requires_human: bool = Field(
        description="True ONLY if there is a legal/SLA threat, suspected hack/2FA takeover, or a billing dispute over $100."
    )
    short_summary: str = Field(
        description="Concise one-sentence summary of the core issue under 15 words."
    )


SYSTEM_PROMPT = """You are an expert customer support triage classifier. Return valid JSON matching the schema.

Rules:
1. Detect sarcasm accurately (e.g., "Oh great, another mystery fee" is angry, not positive).
2. Set `requires_human` to true ONLY when at least one trigger applies:
   - Explicit threat of legal action, lawyers, or SLA damages.
   - Suspected account hack or unauthorized 2FA/security change (NOT a user who lost or broke their own phone).
   - Billing refund or chargeback dispute strictly over $100.
3. For multi-intent tickets, classify by the primary broken bug or urgent issue.

Examples:
- Ticket: "Someone logged in from Russia and changed my recovery email!"
  -> {"category": "account_access", "urgency": "high", "sentiment": "angry", "requires_human": true, "short_summary": "Customer reports unauthorized login and recovery email change."}
- Ticket: "I lost my phone at the airport and need to reset my 2FA app."
  -> {"category": "account_access", "urgency": "medium", "sentiment": "neutral", "requires_human": false, "short_summary": "Customer requests 2FA reset after losing their phone."}
- Ticket: "Why was I billed $12 for extra storage? Please remove it."
  -> {"category": "billing", "urgency": "medium", "sentiment": "neutral", "requires_human": false, "short_summary": "Customer requests removal of a $12 storage charge."}
"""


@retry(
    retry=retry_if_exception_type((ValueError, ValidationError, RuntimeError, Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def classify_ticket(ticket_text: str) -> tuple[TicketAnalysis, float]:
    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"Classify this support ticket:\n\n{ticket_text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=TicketAnalysis,
        ),
    )
    if not response.text:
        raise ValueError("Empty response returned from Gemini model.")
    parsed = TicketAnalysis.model_validate_json(response.text)
    latency_ms = (time.perf_counter() - start) * 1000
    return parsed, latency_ms


def main() -> None:
    results: list[dict] = []
    latencies: list[float] = []
    succeeded = 0

    for idx, ticket in enumerate(SAMPLE_TICKETS, start=1):
        try:
            analysis, latency_ms = classify_ticket(ticket)
            latencies.append(latency_ms)
            succeeded += 1
            results.append({"ticket_id": idx, "status": "ok", "raw_text": ticket, **analysis.model_dump()})
            print(
                f"#{idx:02d} | {analysis.category:<15} | {analysis.urgency:<6} | "
                f"{analysis.sentiment:<8} | human={str(analysis.requires_human):<5} | {latency_ms:5.0f}ms"
            )
        except Exception as exc:
            results.append({"ticket_id": idx, "status": "error", "raw_text": ticket, "error": str(exc)})
            print(f"#{idx:02d} | ERROR           | -      | -        | human=False | {exc}")

    output_path = "classified_tickets.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_ms = (sum(latencies) / len(latencies)) if latencies else 0.0
    print(f"\nSaved {succeeded}/{len(SAMPLE_TICKETS)} classified tickets to {output_path} (Avg latency: {avg_ms:.0f}ms)")

if __name__ == "__main__":
    main()