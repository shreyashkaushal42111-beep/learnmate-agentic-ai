"""
rag_utils.py — Retrieval-Augmented Generation utilities for LearnMate.

Loads the course knowledge base from data/courses.json, embeds topic
descriptions using a local sentence-transformer model, stores them in a
persistent ChromaDB collection, and exposes a retrieval function used by
the agent to ground roadmap generation in real course content.
"""

import json
import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COURSES_PATH = os.path.join(os.path.dirname(__file__), "data", "courses.json")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "learnmate_courses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs locally

_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedder: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------------
# Lazy initialisation helpers
# ---------------------------------------------------------------------------

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Knowledge-base ingestion
# ---------------------------------------------------------------------------

def _load_courses() -> dict:
    with open(COURSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_documents(courses: dict) -> tuple[list[str], list[str], list[dict]]:
    """
    Flatten the courses JSON into (ids, texts, metadatas) for ChromaDB.

    Each document represents one topic bullet from a domain/level combination.
    The metadata carries domain, level, and an optional project hint so the
    agent can reference concrete examples when building a roadmap.
    """
    ids, texts, metadatas = [], [], []
    for domain, domain_data in courses["domains"].items():
        description = domain_data.get("description", "")
        levels = domain_data.get("levels", {})
        projects = domain_data.get("projects", {})

        for level, topics in levels.items():
            level_projects = projects.get(level, [])
            project_hint = "; ".join(level_projects) if level_projects else ""

            for idx, topic in enumerate(topics):
                doc_id = f"{domain}::{level}::{idx}"
                text = (
                    f"Domain: {domain}. "
                    f"Level: {level}. "
                    f"Topic: {topic}. "
                    f"Context: {description}"
                )
                metadata = {
                    "domain": domain,
                    "level": level,
                    "topic": topic,
                    "project_hint": project_hint,
                }
                ids.append(doc_id)
                texts.append(text)
                metadatas.append(metadata)

    return ids, texts, metadatas


def build_index(force_rebuild: bool = False) -> None:
    """
    Build (or rebuild) the ChromaDB vector index from courses.json.

    Called once on startup; subsequent calls are no-ops unless
    force_rebuild=True.
    """
    collection = _get_collection()

    if not force_rebuild and collection.count() > 0:
        return  # already indexed

    # Drop existing docs on forced rebuild
    if force_rebuild and collection.count() > 0:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    courses = _load_courses()
    ids, texts, metadatas = _build_documents(courses)
    embedder = _get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # Upsert in batches of 100 to stay within ChromaDB limits
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_topics(
    domain: str,
    level: str,
    query: str = "",
    top_k: int = 20,
) -> list[dict]:
    """
    Retrieve the most relevant course topics for a given domain and level.

    Parameters
    ----------
    domain  : e.g. "Frontend Development"
    level   : "beginner" | "intermediate" | "advanced"
    query   : optional free-text query to bias retrieval (e.g. student interest)
    top_k   : maximum number of results to return

    Returns
    -------
    List of dicts with keys: topic, domain, level, project_hint
    """
    collection = _get_collection()

    if collection.count() == 0:
        build_index()

    # Build a rich query string that incorporates domain + level context
    effective_query = (
        f"{query} {domain} {level} course topics"
        if query
        else f"{domain} {level} learning topics"
    )

    embedder = _get_embedder()
    query_embedding = embedder.encode(effective_query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where={"$and": [{"domain": domain}, {"level": level}]},
        include=["metadatas", "documents"],
    )

    topics = []
    seen = set()
    for meta in results["metadatas"][0]:
        topic = meta["topic"]
        if topic not in seen:
            seen.add(topic)
            topics.append(
                {
                    "topic": topic,
                    "domain": meta["domain"],
                    "level": meta["level"],
                    "project_hint": meta.get("project_hint", ""),
                }
            )

    return topics


def retrieve_topics_for_roadmap(
    domain: str,
    level: str,
    num_weeks: int = 8,
) -> dict:
    """
    Return a structured dict of topics chunked into weekly groups,
    ready for the agent to use as RAG context when generating a roadmap.

    Returns
    -------
    {
        "domain": str,
        "level": str,
        "topics": [str, ...],          # all retrieved topics
        "weekly_chunks": [[str, ...],] # topics split across weeks
        "project_suggestions": [str]   # project hints from metadata
    }
    """
    results = retrieve_topics(domain, level, top_k=50)

    topics = [r["topic"] for r in results]
    project_hints_raw = set()
    for r in results:
        for hint in r["project_hint"].split(";"):
            hint = hint.strip()
            if hint:
                project_hints_raw.add(hint)

    # Distribute topics evenly across weeks
    topics_per_week = max(1, len(topics) // num_weeks)
    weekly_chunks = [
        topics[i : i + topics_per_week]
        for i in range(0, len(topics), topics_per_week)
    ]

    return {
        "domain": domain,
        "level": level,
        "topics": topics,
        "weekly_chunks": weekly_chunks[:num_weeks],
        "project_suggestions": list(project_hints_raw),
    }


def get_available_domains() -> list[str]:
    """Return the list of domains present in the knowledge base."""
    courses = _load_courses()
    return list(courses["domains"].keys())


def get_domain_description(domain: str) -> str:
    """Return the description for a given domain."""
    courses = _load_courses()
    return courses["domains"].get(domain, {}).get("description", "")
