from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel
import json
from pathlib import Path

load_dotenv()

app = FastAPI(title="Gemini Tool Calling + FastAPI")
client = genai.Client()  # GEMINI_API_KEY-ah automatic-aa eduthukkum


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    used_tool: bool


def get_weather(city: str) -> str:
    """Oru city-oda current weather-ah weather.json file-la irundhu eduthu tharum.

    Args:
        city: City peyar (eg: Chennai, Pudukkottai, Bengaluru)
    """
    file_path = Path(__file__).parent / "weather.json"
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return data.get(city, f"{city}-kku weather data kidaikkala")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_with_tool(req: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=req.message,
            config=types.GenerateContentConfig(
                tools=[get_weather],
            ),
        )

        # Tool call nadandhirukka nu check panrom
        used_tool = False
        for part in response.candidates[0].content.parts:
            if part.function_call:
                used_tool = True
                break

        return ChatResponse(
            reply=response.text or "",
            used_tool=used_tool,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))