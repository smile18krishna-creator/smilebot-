import json
import os
import time
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
REFUSAL_TEXT = "Insufficient context in the provided documents."
DISTANCE_THRESHOLD = 0.55

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

client = genai.Client(api_key=API_KEY)

TEST_QUESTIONS = [
    "What is the maximum home office setup stipend, and within how many days of joining must receipts be submitted?",
    "When is an employee required to provide a doctor's medical certificate for sick leave?",
    "What is the daily meal reimbursement cap during client travel, and what items are strictly excluded?",
    "How many unused annual paid leave days can carry over to the next year, and what happens to any extra days?",
    "If an employee buys a monitor and also claims home broadband, what are the reimbursement caps and submission timelines for each?",
    "How many weeks of paid maternity or paternity leave do full-time employees receive?",
    "What is the reimbursement limit for international business-class flight upgrades?",
    "Are employees allowed to work remotely from outside India for up to 30 days per year?",
]


class RAGResponse(BaseModel):
    answer: str = Field(
        description=f"Grounded answer with inline [Page X] citations, or exact string '{REFUSAL_TEXT}' if unsupported."
    )
    cited_pages: list[int] = Field(
        description="List of unique integer page numbers cited in the answer. Empty list if refused."
    )
    is_grounded: bool = Field(
        description="True if the provided context explicitly answers the question; False if refused."
    )


SYSTEM_PROMPT = f"""
You are a strict company policy compliance assistant. Answer the user's question using ONLY the retrieved context blocks below.

Mandatory Rules:
1. Do NOT use outside knowledge, labor law defaults, or assumptions.
2. Every factual statement in your answer MUST include an inline page citation like [Page 1] or [Page 3].
3. If the context does not explicitly contain the answer to ANY part of the user's question, set `is_grounded` to false, `cited_pages` to [], and set `answer` to the exact string:
"{REFUSAL_TEXT}"
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def get_embedding(text: str) -> list[float]:
    res = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return res.embeddings[0].values


def retrieve_context(collection, question: str, top_k: int = 4) -> tuple[str, float]:
    query_vec = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    min_distance = min(dists) if dists else 1.0
    blocks = []
    for doc_text, meta, dist in zip(docs, metas, dists):
        blocks.append(f"[Source: {meta['source']} | Page {meta['page']} | dist={dist:.4f}]\n{doc_text}")

    return "\n\n---\n\n".join(blocks), min_distance


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def generate_grounded_reply(question: str, context_block: str) -> RAGResponse:
    prompt = f"RETRIEVED CONTEXT:\n{context_block}\n\nUSER QUESTION:\n{question}"
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RAGResponse,
        ),
    )
    return RAGResponse.model_validate_json(response.text)


def answer_question(collection, question: str) -> tuple[RAGResponse, float, float]:
    start = time.perf_counter()
    context_block, min_dist = retrieve_context(collection, question, top_k=4)

    # Layer 1: Distance threshold check before LLM generation
    if min_dist > DISTANCE_THRESHOLD:
        latency_ms = (time.perf_counter() - start) * 1000
        refusal = RAGResponse(answer=REFUSAL_TEXT, cited_pages=[], is_grounded=False)
        return refusal, min_dist, latency_ms

    # Layer 2: Strict grounded generation at temperature=0.0
    rag_res = generate_grounded_reply(question, context_block)
    if not rag_res.is_grounded or REFUSAL_TEXT.lower() in rag_res.answer.lower():
        rag_res = RAGResponse(answer=REFUSAL_TEXT, cited_pages=[], is_grounded=False)

    latency_ms = (time.perf_counter() - start) * 1000
    return rag_res, min_dist, latency_ms


def main():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="company_policy",
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        raise RuntimeError("Collection 'company_policy' is empty. Run 03_pdf_search.py first.")

    outputs = []
    for idx, q in enumerate(TEST_QUESTIONS, start=1):
        rag_res, min_dist, latency_ms = answer_question(collection, q)
        outputs.append(
            {
                "q_id": idx,
                "question": q,
                "min_distance": round(min_dist, 4),
                "latency_ms": round(latency_ms, 1),
                **rag_res.model_dump(),
            }
        )
        status = "GROUNDED" if rag_res.is_grounded else "REFUSED "
        print(f"Q{idx} [{status} | dist={min_dist:.3f} | pages={rag_res.cited_pages} | {latency_ms:5.0f}ms]")
        print(f"   {rag_res.answer}\n")

    with open("rag_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print("Saved 8 evaluation results to rag_eval_results.json")


if __name__ == "__main__":
    main()
