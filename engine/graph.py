"""
Graph builder — builds sparse matrices for collaborative filtering.

Core idea:
  user×repo star matrix → repo×repo co-occurrence matrix
  similarity(A, B) = number of users who starred both A and B
"""

import csv
from pathlib import Path

import numpy as np
from scipy import sparse

DATA_DIR = Path(__file__).parent / "data"
STARS_FILE = DATA_DIR / "stars.csv"
GRAPH_FILE = DATA_DIR / "graph.npz"
INDEX_FILE = DATA_DIR / "index.json"


def load_stars() -> list[tuple[str, str]]:
    """Load (user, repo) pairs from CSV."""
    rows = []
    with open(STARS_FILE) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0], row[1]))
    print(f"Loaded {len(rows):,} star edges")
    return rows


def build_matrices(stars: list[tuple[str, str]]):
    """
    Build sparse matrices from star edges.

    Returns:
        user_repo: sparse matrix (users × repos), 1 where user starred repo
        repo_repo: sparse matrix (repos × repos), co-occurrence counts
        repo_index: dict repo_name → column index
        repo_names: list of repo names (index → name)
    """
    # Build index mappings
    users = sorted(set(s[0] for s in stars))
    repos = sorted(set(s[1] for s in stars))

    user_to_idx = {u: i for i, u in enumerate(users)}
    repo_to_idx = {r: i for i, r in enumerate(repos)}

    print(f"Building matrix: {len(users):,} users × {len(repos):,} repos")

    # Build user×repo sparse matrix
    row_idx = []
    col_idx = []
    for user, repo in stars:
        row_idx.append(user_to_idx[user])
        col_idx.append(repo_to_idx[repo])

    data = np.ones(len(row_idx), dtype=np.float32)
    user_repo = sparse.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(users), len(repos)),
    )

    # Deduplicate: if user starred repo multiple times, cap at 1
    user_repo.data[:] = 1.0

    print(f"User×Repo matrix: {user_repo.shape}, {user_repo.nnz:,} non-zeros")

    # Build repo×repo co-occurrence: R^T @ R
    # Each cell (i,j) = number of users who starred both repo_i and repo_j
    print("Computing co-occurrence matrix (R^T @ R)...")
    repo_repo = (user_repo.T @ user_repo).tocsr()

    # Zero out diagonal (repo is always similar to itself)
    repo_repo.setdiag(0)
    repo_repo.eliminate_zeros()

    print(f"Repo×Repo matrix: {repo_repo.shape}, {repo_repo.nnz:,} non-zeros")

    # Normalize to cosine similarity
    # cosine(A,B) = co_stars(A,B) / sqrt(stars(A) * stars(B))
    star_counts = np.array(user_repo.sum(axis=0)).flatten()  # stars per repo
    star_counts = np.maximum(star_counts, 1)  # avoid div by zero
    norm = np.sqrt(star_counts)

    # Normalize: divide each row by norm[row], each col by norm[col]
    norm_matrix = sparse.diags(1.0 / norm)
    repo_repo_norm = norm_matrix @ repo_repo @ norm_matrix

    return user_repo, repo_repo_norm, repo_to_idx, repos, user_to_idx, users


def save_graph(repo_repo, repo_to_idx, repo_names, user_repo=None):
    """Save graph to disk."""
    import json

    DATA_DIR.mkdir(exist_ok=True)
    sparse.save_npz(GRAPH_FILE, repo_repo)

    index = {
        "repo_to_idx": repo_to_idx,
        "repo_names": repo_names,
    }
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f)

    if user_repo is not None:
        sparse.save_npz(DATA_DIR / "user_repo.npz", user_repo)

    mb = GRAPH_FILE.stat().st_size / 1e6
    print(f"Saved graph to {GRAPH_FILE} ({mb:.1f} MB)")


def load_graph():
    """Load pre-built graph from disk."""
    import json

    repo_repo = sparse.load_npz(GRAPH_FILE)
    with open(INDEX_FILE) as f:
        index = json.load(f)

    repo_to_idx = index["repo_to_idx"]
    repo_names = index["repo_names"]

    print(f"Loaded graph: {len(repo_names):,} repos, {repo_repo.nnz:,} edges")
    return repo_repo, repo_to_idx, repo_names


def get_co_star_recommendations(repo_name: str, repo_repo, repo_to_idx, repo_names, top_n: int = 20):
    """
    Get recommendations based on co-star similarity.

    Returns list of (repo_name, score) sorted by score descending.
    """
    if repo_name not in repo_to_idx:
        return []

    idx = repo_to_idx[repo_name]
    scores = np.array(repo_repo[idx].todense()).flatten()

    # Get top indices
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []
    for i in top_indices:
        if scores[i] > 0:
            results.append((repo_names[i], float(scores[i])))

    return results
