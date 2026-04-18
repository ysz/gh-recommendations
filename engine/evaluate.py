"""
Evaluation — measure recommendation quality.

Metrics:
- Hit Rate @ K: how often a hidden star appears in top-K recs
- NDCG @ K: ranking quality (higher = hidden stars ranked higher)
- Coverage: % of repos that appear in at least one recommendation
"""

import csv
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
STARS_FILE = DATA_DIR / "stars.csv"


def load_user_stars() -> dict[str, set[str]]:
    """Load stars grouped by user."""
    user_stars = {}
    with open(STARS_FILE) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                user, repo = row[0], row[1]
                if user not in user_stars:
                    user_stars[user] = set()
                user_stars[user].add(repo)
    return user_stars


def evaluate_hit_rate(recommender, k: int = 20, n_users: int = 500, hide_n: int = 3, seed: int = 42):
    """
    Held-out evaluation.

    For each test user:
    1. Hide `hide_n` random starred repos
    2. Get recommendations based on remaining stars
    3. Check if hidden repos appear in top-K
    """
    random.seed(seed)
    user_stars = load_user_stars()

    # Filter users with enough stars
    eligible = {u: repos for u, repos in user_stars.items() if len(repos) >= hide_n + 5}
    if not eligible:
        print("Not enough users with sufficient stars for evaluation")
        return {}

    test_users = random.sample(list(eligible.keys()), min(n_users, len(eligible)))

    hits = 0
    total = 0
    reciprocal_ranks = []

    for user in tqdm(test_users, desc="Evaluating"):
        all_stars = list(eligible[user])
        random.shuffle(all_stars)

        hidden = set(all_stars[:hide_n])
        visible = [r for r in all_stars if r not in hidden]

        recs = recommender.recommend_for_user(visible, top_n=k)
        rec_names = [r["repo"] for r in recs]

        # Hit rate: did any hidden repo appear?
        found = hidden & set(rec_names)
        if found:
            hits += 1
            # Reciprocal rank of first hit
            for i, name in enumerate(rec_names):
                if name in hidden:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break

        total += 1

    hit_rate = hits / total if total > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    results = {
        "hit_rate_at_k": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "k": k,
        "n_users_tested": total,
        "hide_n": hide_n,
    }

    print(f"\nEvaluation Results:")
    print(f"  Hit Rate @ {k}: {hit_rate:.1%}")
    print(f"  MRR:            {mrr:.4f}")
    print(f"  Users tested:   {total}")
    print(f"  Hidden per user: {hide_n}")

    return results
