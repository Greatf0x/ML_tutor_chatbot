import chromadb
from chromadb.config import Settings

from src.rag.embedder import Embedder

# ── Tuning knob ──────────────────────────────────────────────────────────────
# ChromaDB uses L2 distance: 0 = identical, higher = less similar.
# Chunks whose distance is ABOVE this value are treated as "not in the notes".
# Lower  → stricter  (fewer answers, safer)
# Higher → looser    (more answers, risks hallucination)
SIMILARITY_THRESHOLD = 1.2  # tuned for all-MiniLM-L6-v2 normalized vectors (L2 range 0–2)
# ─────────────────────────────────────────────────────────────────────────────


class SimpleRetriever:
    def __init__(self, collection_name: str = "ml_tutor_notes") -> None:
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = Embedder()

    def add_chunks(self, chunks: list[str]) -> None:
        if not chunks:
            return

        existing = self.collection.get()
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

        embeddings = self.embedder.encode_documents(chunks)
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_embedding = self.embedder.encode_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],
        )

        docs = results.get("documents", [[]])
        distances = results.get("distances", [[]])

        print("\nDEBUG QUERY:", query)
        print("DEBUG DISTANCES:", distances)

        if not docs or not docs[0]:
            return []

        # ── NEW: filter out chunks that are too dissimilar ────────────────────
        filtered = [
            doc
            for doc, dist in zip(docs[0], distances[0])
            if dist <= SIMILARITY_THRESHOLD
        ]

        print(f"DEBUG: {len(docs[0])} chunks retrieved, "
              f"{len(filtered)} passed threshold ({SIMILARITY_THRESHOLD})")

        return filtered