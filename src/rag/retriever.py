import re

import chromadb
from chromadb.config import Settings

from src.rag.embedder import Embedder


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


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

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        max_distance: float = 1.20,
        min_keyword_overlap: int = 1,
    ) -> list[str]:
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

        query_words = set(normalize_text(query).split())
        filtered_docs: list[str] = []

        for doc, distance in zip(docs[0], distances[0]):
            normalized_doc = normalize_text(doc)
            doc_words = set(normalized_doc.split())
            overlap = len(query_words & doc_words)

            print("\nDOC PREVIEW:", doc[:250])
            print("DISTANCE:", distance)
            print("KEYWORD OVERLAP:", overlap)

            if distance is not None and distance <= max_distance:
                filtered_docs.append(doc)
            elif overlap >= min_keyword_overlap:
                filtered_docs.append(doc)

        return filtered_docs