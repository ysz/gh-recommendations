"""
Data collector for GitHub repository recommendation graph.

Two modes:
1. GitHub API (prototype) — collect neighborhood around seed repos
2. BigQuery CSV import (production) — load pre-exported star data
"""

import csv
import json
import os
import time
from pathlib import Path

import httpx
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
STARS_FILE = DATA_DIR / "stars.csv"
REPOS_FILE = DATA_DIR / "repos.csv"

GITHUB_API = "https://api.github.com"


def get_client(token: str | None = None) -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=GITHUB_API, headers=headers, timeout=30)


def check_rate_limit(client: httpx.Client) -> dict:
    r = client.get("/rate_limit")
    r.raise_for_status()
    core = r.json()["resources"]["core"]
    return core


def wait_for_rate_limit(client: httpx.Client):
    info = check_rate_limit(client)
    if info["remaining"] < 5:
        wait = max(0, info["reset"] - time.time()) + 2
        print(f"  Rate limit hit. Waiting {wait:.0f}s...")
        time.sleep(wait)


def fetch_stargazers(client: httpx.Client, repo: str, max_pages: int = 5) -> list[str]:
    """Fetch usernames who starred a repo. Returns up to max_pages * 100 users."""
    users = []
    for page in range(1, max_pages + 1):
        wait_for_rate_limit(client)
        r = client.get(f"/repos/{repo}/stargazers", params={"per_page": 100, "page": page})
        if r.status_code != 200:
            break
        batch = [u["login"] for u in r.json()]
        if not batch:
            break
        users.extend(batch)
    return users


def fetch_user_stars(client: httpx.Client, username: str, max_pages: int = 3) -> list[dict]:
    """Fetch repos starred by a user. Returns repo metadata."""
    repos = []
    for page in range(1, max_pages + 1):
        wait_for_rate_limit(client)
        r = client.get(
            f"/users/{username}/starred",
            params={"per_page": 100, "page": page},
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        for repo in batch:
            repos.append({
                "full_name": repo["full_name"],
                "description": (repo.get("description") or "")[:500],
                "language": repo.get("language") or "",
                "stars": repo.get("stargazers_count", 0),
                "topics": ",".join(repo.get("topics", [])),
            })
    return repos


def collect_neighborhood(
    seed_repos: list[str],
    token: str | None = None,
    stargazers_per_repo: int = 100,
    stars_per_user: int = 200,
):
    """
    Collect star graph around seed repos via GitHub API.

    1. For each seed repo, fetch sample of stargazers
    2. For each stargazer, fetch their starred repos
    3. Save to CSV files
    """
    client = get_client(token)

    rate = check_rate_limit(client)
    print(f"Rate limit: {rate['remaining']}/{rate['limit']} remaining")
    if not token and rate["limit"] <= 60:
        print("WARNING: No GitHub token. Rate limit is 60/hour.")
        print("Set GITHUB_TOKEN env var for 5000/hour.")
        print()

    stars_rows = []  # (user, repo)
    repos_meta = {}  # repo -> metadata

    stargazer_pages = max(1, stargazers_per_repo // 100)
    star_pages = max(1, stars_per_user // 100)

    all_stargazers = set()

    for seed in seed_repos:
        print(f"\nCollecting stargazers of {seed}...")
        gazers = fetch_stargazers(client, seed, max_pages=stargazer_pages)
        print(f"  Got {len(gazers)} stargazers")
        all_stargazers.update(gazers)

        # Record that each gazer starred the seed
        for user in gazers:
            stars_rows.append((user, seed))

    print(f"\nCollecting starred repos for {len(all_stargazers)} users...")
    for user in tqdm(all_stargazers, desc="Users"):
        user_repos = fetch_user_stars(client, user, max_pages=star_pages)
        for repo in user_repos:
            stars_rows.append((user, repo["full_name"]))
            if repo["full_name"] not in repos_meta:
                repos_meta[repo["full_name"]] = repo

    # Save stars
    DATA_DIR.mkdir(exist_ok=True)
    with open(STARS_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user", "repo"])
        for row in stars_rows:
            w.writerow(row)

    # Save repo metadata
    with open(REPOS_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["full_name", "description", "language", "stars", "topics"])
        for repo in repos_meta.values():
            w.writerow([
                repo["full_name"],
                repo["description"],
                repo["language"],
                repo["stars"],
                repo["topics"],
            ])

    print(f"\nDone!")
    print(f"  Stars (edges): {len(stars_rows):,}")
    print(f"  Repos (nodes): {len(repos_meta):,}")
    print(f"  Users (nodes): {len(all_stargazers):,}")
    print(f"  Saved to {STARS_FILE} and {REPOS_FILE}")

    client.close()
    return stars_rows, repos_meta


def load_bigquery_export(stars_csv: str, repos_csv: str | None = None):
    """
    Load pre-exported BigQuery data.

    Expected format for stars_csv: user,repo (no header or with header)
    Expected format for repos_csv: full_name,description,language,stars,topics
    """
    import shutil
    DATA_DIR.mkdir(exist_ok=True)

    src = Path(stars_csv)
    if src != STARS_FILE:
        shutil.copy2(src, STARS_FILE)
        print(f"Copied {src} → {STARS_FILE}")

    if repos_csv:
        src = Path(repos_csv)
        if src != REPOS_FILE:
            shutil.copy2(src, REPOS_FILE)
            print(f"Copied {src} → {REPOS_FILE}")

    # Count
    with open(STARS_FILE) as f:
        lines = sum(1 for _ in f) - 1
    print(f"Loaded {lines:,} star edges")


# BigQuery SQL for production-scale data collection
BIGQUERY_SQL = """
-- Run this in Google BigQuery (console.cloud.google.com/bigquery)
-- Free tier: 1 TB/month scanning

-- Step 1: Get all star events for repos with 100+ stars (last 6 months)
-- Adjust date range and star threshold as needed

CREATE TEMP TABLE active_repos AS
SELECT repo.name as repo_name, COUNT(*) as star_count
FROM `githubarchive.month.202601`,
     `githubarchive.month.202602`,
     `githubarchive.month.202603`,
     `githubarchive.month.202604`
WHERE type = 'WatchEvent'
GROUP BY repo.name
HAVING star_count >= 100;

-- Step 2: Get all star edges for these repos
SELECT actor.login as user, repo.name as repo
FROM `githubarchive.month.202601`,
     `githubarchive.month.202602`,
     `githubarchive.month.202603`,
     `githubarchive.month.202604`
WHERE type = 'WatchEvent'
  AND repo.name IN (SELECT repo_name FROM active_repos)
ORDER BY user, repo;

-- Export result as CSV → load with:
-- python main.py load-bigquery exported_stars.csv
"""
