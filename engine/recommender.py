"""
Hybrid recommendation engine.

Combines multiple signals:
- Co-star similarity (collaborative filtering)
- Topic overlap (content-based)
- Description embeddings (semantic similarity)
"""

import csv
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
REPOS_FILE = DATA_DIR / "repos.csv"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
EMBEDDINGS_INDEX_FILE = DATA_DIR / "embeddings_index.json"


def load_repo_metadata() -> dict[str, dict]:
    """Load repo metadata from CSV."""
    repos = {}
    if not REPOS_FILE.exists():
        return repos
    with open(REPOS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            repos[row["full_name"]] = {
                "description": row.get("description", ""),
                "language": row.get("language", ""),
                "stars": int(row.get("stars", 0)),
                "topics": row.get("topics", "").split(",") if row.get("topics") else [],
            }
    return repos


def compute_topic_overlap(topics_a: list[str], topics_b: list[str]) -> float:
    """Jaccard similarity between topic sets."""
    if not topics_a or not topics_b:
        return 0.0
    set_a, set_b = set(topics_a), set(topics_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def build_embeddings(repo_metadata: dict[str, dict], batch_size: int = 256):
    """
    Build description embeddings using sentence-transformers.
    Runs locally on CPU, ~5 min for 144K repos.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    repo_names = sorted(repo_metadata.keys())
    texts = []
    for name in repo_names:
        meta = repo_metadata[name]
        # Combine description + topics for richer embedding
        parts = []
        if meta["description"]:
            parts.append(meta["description"])
        if meta["topics"]:
            parts.append("Topics: " + ", ".join(meta["topics"]))
        if meta["language"]:
            parts.append(f"Language: {meta['language']}")
        texts.append(" ".join(parts) if parts else name)

    print(f"Computing embeddings for {len(texts)} repos...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms

    np.save(EMBEDDINGS_FILE, embeddings.astype(np.float32))
    with open(EMBEDDINGS_INDEX_FILE, "w") as f:
        json.dump(repo_names, f)

    print(f"Saved embeddings: {embeddings.shape} → {EMBEDDINGS_FILE}")
    return embeddings, repo_names


def load_embeddings():
    """Load pre-computed embeddings."""
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(EMBEDDINGS_INDEX_FILE) as f:
        repo_names = json.load(f)
    return embeddings, repo_names


class Recommender:
    """Hybrid recommendation engine combining multiple signals."""

    def __init__(
        self,
        repo_repo,
        repo_to_idx: dict,
        repo_names: list,
        repo_metadata: dict | None = None,
        embeddings: np.ndarray | None = None,
        embedding_names: list | None = None,
    ):
        self.repo_repo = repo_repo
        self.repo_to_idx = repo_to_idx
        self.repo_names = repo_names
        self.metadata = repo_metadata or {}
        self.embeddings = embeddings
        self.embedding_to_idx = {}
        if embedding_names:
            self.embedding_to_idx = {n: i for i, n in enumerate(embedding_names)}

    def recommend(
        self,
        repo_name: str,
        top_n: int = 20,
        min_stars: int = 0,
        w_costars: float = 0.4,
        w_embedding: float = 0.35,
        w_topics: float = 0.15,
        w_deps: float = 0.1,
    ) -> list[dict]:
        """
        Get hybrid recommendations for a repo.

        Returns list of dicts with repo info and scores.
        """
        candidates = set()
        scores_costars = {}
        scores_embedding = {}
        scores_topics = {}

        # --- Signal 1: Co-star similarity ---
        if repo_name in self.repo_to_idx:
            idx = self.repo_to_idx[repo_name]
            row = np.array(self.repo_repo[idx].todense()).flatten()
            top_idx = np.argsort(row)[::-1][:top_n * 5]
            for i in top_idx:
                if row[i] > 0:
                    name = self.repo_names[i]
                    scores_costars[name] = float(row[i])
                    candidates.add(name)

        # --- Signal 2: Embedding similarity ---
        if self.embeddings is not None and repo_name in self.embedding_to_idx:
            idx = self.embedding_to_idx[repo_name]
            query_vec = self.embeddings[idx]
            sims = self.embeddings @ query_vec  # cosine sim (already normalized)
            top_idx = np.argsort(sims)[::-1][:top_n * 5]
            emb_names = list(self.embedding_to_idx.keys())
            for i in top_idx:
                if i < len(emb_names):
                    name = emb_names[i]
                    if name != repo_name:
                        scores_embedding[name] = float(sims[i])
                        candidates.add(name)

        # --- Signal 3: Topic overlap ---
        query_meta = self.metadata.get(repo_name, {})
        query_topics = query_meta.get("topics", [])
        if query_topics:
            for name in candidates:
                meta = self.metadata.get(name, {})
                scores_topics[name] = compute_topic_overlap(
                    query_topics, meta.get("topics", [])
                )

        # --- Combine scores ---
        if not candidates:
            return []

        # Normalize each signal to [0, 1]
        def normalize(d: dict) -> dict:
            if not d:
                return d
            max_val = max(d.values())
            if max_val > 0:
                return {k: v / max_val for k, v in d.items()}
            return d

        scores_costars = normalize(scores_costars)
        scores_embedding = normalize(scores_embedding)
        scores_topics = normalize(scores_topics)

        # Weighted combination
        final_scores = {}
        for name in candidates:
            if name == repo_name:
                continue
            # Skip repos with too few stars
            meta = self.metadata.get(name, {})
            if meta.get("stars", 0) < min_stars:
                continue
            score = (
                scores_costars.get(name, 0) * w_costars
                + scores_embedding.get(name, 0) * w_embedding
                + scores_topics.get(name, 0) * w_topics
            )
            final_scores[name] = score

        # Sort and return top_n
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])[:top_n]

        results = []
        for name, score in ranked:
            meta = self.metadata.get(name, {})
            results.append({
                "repo": name,
                "score": round(score, 4),
                "stars": meta.get("stars", 0),
                "language": meta.get("language", ""),
                "description": meta.get("description", "")[:120],
                "topics": meta.get("topics", []),
                "signals": {
                    "co_stars": round(scores_costars.get(name, 0), 3),
                    "embedding": round(scores_embedding.get(name, 0), 3),
                    "topics": round(scores_topics.get(name, 0), 3),
                },
            })

        return results

    def recommend_for_user(
        self, starred_repos: list[str], top_n: int = 20, **kwargs
    ) -> list[dict]:
        """
        Recommend repos for a user based on their starred repos.
        Aggregates recommendations across all starred repos.
        """
        all_scores = {}

        for repo in starred_repos:
            recs = self.recommend(repo, top_n=top_n * 2, **kwargs)
            for rec in recs:
                name = rec["repo"]
                if name in starred_repos:
                    continue  # don't recommend already-starred
                if name not in all_scores:
                    all_scores[name] = rec
                    all_scores[name]["_total"] = rec["score"]
                    all_scores[name]["_count"] = 1
                else:
                    all_scores[name]["_total"] += rec["score"]
                    all_scores[name]["_count"] += 1

        # Score = average relevance × number of starred repos that led here
        for name, rec in all_scores.items():
            rec["score"] = round(rec["_total"] * (1 + 0.2 * (rec["_count"] - 1)), 4)
            del rec["_total"]
            del rec["_count"]

        ranked = sorted(all_scores.values(), key=lambda x: -x["score"])[:top_n]
        return ranked
