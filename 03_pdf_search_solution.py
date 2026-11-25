import os
import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

client = genai.Client(api_key=API_KEY)

TEST_QUERIES = [
    "When do I need a doctor's certificate for sick leave?",
    "How much can I claim for home internet and how often?",
    "What happens to unused annual leave at the end of the year?",
    "How quickly must I return my laptop after resigning?",
    "Are drinks or cigarettes covered during client travel meals?",
]


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 70) -> list[str]:
    assert overlap < chunk_size, "overlap must be smaller than chunk_size"
    chunks = []
    start = 0
    cleaned = " ".join(text.split())
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        start += chunk_size - overlap
    return chunks


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def index_pdf(pdf_path: str, collection, chunk_size: int = 350, overlap: int = 70):
    doc = fitz.open(pdf_path)
    ids, documents, metadatas, embeddings = [], [], [], []

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        page_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(page_chunks):
            chunk_id = f"{pdf_path}_p{page_num}_c{chunk_idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source": pdf_path, "page": page_num})
            embeddings.append(get_embedding(chunk))

    doc.close()
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def search_policy(collection, query: str, top_k: int = 3):
    query_vec = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for rank, (doc_text, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        preview = doc_text[:110] + ("..." if len(doc_text) > 110 else "")
        print(f"  #{rank} [Page {meta['page']} | dist={dist:.4f}]: {preview}")


def main():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="company_policy",
        metadata={"hnsw:space": "cosine"},
    )

    index_pdf("company_policy.pdf", collection, chunk_size=350, overlap=70)
    print(f"Total chunks indexed in ChromaDB: {collection.count()}\n")

    for q in TEST_QUERIES:
        print(f"Query: {q}")
        search_policy(collection, q, top_k=3)
        print("-" * 60)


if __name__ == "__main__":
    main()
