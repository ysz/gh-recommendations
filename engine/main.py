#!/usr/bin/env python3
"""
GitHub Repository Recommendation System

Usage:
  python main.py collect facebook/react vuejs/vue   # collect data around seed repos
  python main.py build                               # build graph from collected data
  python main.py embeddings                           # compute description embeddings
  python main.py recommend facebook/react             # get recommendations
  python main.py recommend-user USERNAME              # recommendations for a GitHub user
  python main.py evaluate                             # evaluate quality
  python main.py load-bigquery stars.csv [repos.csv]  # load BigQuery export
  python main.py bigquery-sql                         # print BigQuery SQL
"""

import os
import sys

import click


@click.group()
def cli():
    """GitHub Repository Recommendation System"""
    pass


@cli.command()
@click.argument("seed_repos", nargs=-1, required=True)
@click.option("--stargazers", default=100, help="Stargazers to sample per seed repo")
@click.option("--stars-per-user", default=200, help="Max stars to fetch per user")
@click.option("--token", envvar="GITHUB_TOKEN", default=None, help="GitHub API token")
def collect(seed_repos, stargazers, stars_per_user, token):
    """Collect star graph around seed repositories."""
    from collector import collect_neighborhood
    collect_neighborhood(
        list(seed_repos),
        token=token,
        stargazers_per_repo=stargazers,
        stars_per_user=stars_per_user,
    )


@cli.command()
def build():
    """Build graph matrices from collected data."""
    from graph import build_matrices, load_stars, save_graph
    stars = load_stars()
    user_repo, repo_repo, repo_to_idx, repo_names, _, _ = build_matrices(stars)
    save_graph(repo_repo, repo_to_idx, repo_names, user_repo)


@cli.command()
@click.option("--batch-size", default=256, help="Batch size for embedding model")
def embeddings(batch_size):
    """Compute description embeddings for repos."""
    from recommender import build_embeddings, load_repo_metadata
    metadata = load_repo_metadata()
    if not metadata:
        print("No repo metadata found. Run 'collect' first.")
        return
    build_embeddings(metadata, batch_size=batch_size)


@cli.command()
@click.argument("repo_name")
@click.option("--top", default=20, help="Number of recommendations")
@click.option("--no-embeddings", is_flag=True, help="Skip embedding signal")
def recommend(repo_name, top, no_embeddings):
    """Get recommendations for a repository."""
    rec = _build_recommender(use_embeddings=not no_embeddings)

    results = rec.recommend(repo_name, top_n=top)
    if not results:
        print(f"Repository '{repo_name}' not found in graph.")
        print("Make sure you've collected data and built the graph.")
        return

    print(f"\nRecommendations for {repo_name}:\n")
    print(f"{'#':<4} {'Repository':<40} {'Score':<8} {'Stars':<8} {'Lang':<12} Description")
    print("-" * 120)
    for i, r in enumerate(results, 1):
        print(
            f"{i:<4} {r['repo']:<40} {r['score']:<8.3f} "
            f"{r['stars']:<8} {r['language']:<12} {r['description'][:50]}"
        )


@cli.command("recommend-user")
@click.argument("username")
@click.option("--top", default=20, help="Number of recommendations")
@click.option("--token", envvar="GITHUB_TOKEN", default=None)
def recommend_user(username, top, token):
    """Get personalized recommendations for a GitHub user."""
    import httpx

    # Fetch user's stars
    print(f"Fetching stars for {username}...")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    starred = []
    for page in range(1, 11):  # max 1000 stars
        r = httpx.get(
            f"https://api.github.com/users/{username}/starred",
            params={"per_page": 100, "page": page},
            headers=headers,
        )
        if r.status_code != 200:
            break
        batch = [repo["full_name"] for repo in r.json()]
        if not batch:
            break
        starred.extend(batch)

    print(f"Found {len(starred)} starred repos")

    rec = _build_recommender(use_embeddings=True)
    results = rec.recommend_for_user(starred, top_n=top)

    if not results:
        print("No recommendations found. Try collecting more data.")
        return

    print(f"\nPersonalized recommendations for @{username}:\n")
    print(f"{'#':<4} {'Repository':<40} {'Score':<8} {'Stars':<8} {'Lang':<12} Description")
    print("-" * 120)
    for i, r in enumerate(results, 1):
        print(
            f"{i:<4} {r['repo']:<40} {r['score']:<8.3f} "
            f"{r['stars']:<8} {r['language']:<12} {r['description'][:50]}"
        )


@cli.command()
@click.option("--k", default=20, help="Top-K for evaluation")
@click.option("--users", default=500, help="Number of users to test")
def evaluate(k, users):
    """Evaluate recommendation quality with held-out test."""
    from evaluate import evaluate_hit_rate
    rec = _build_recommender(use_embeddings=True)
    evaluate_hit_rate(rec, k=k, n_users=users)


@cli.command("load-bigquery")
@click.argument("stars_csv")
@click.argument("repos_csv", required=False)
def load_bigquery(stars_csv, repos_csv):
    """Load pre-exported BigQuery CSV data."""
    from collector import load_bigquery_export
    load_bigquery_export(stars_csv, repos_csv)
    print("\nNow run: python main.py build")


@cli.command("bigquery-sql")
def bigquery_sql():
    """Print BigQuery SQL for production data collection."""
    from collector import BIGQUERY_SQL
    print(BIGQUERY_SQL)


@cli.command("export-web")
@click.option("--output", default=None, help="Output directory")
@click.option("--top", default=10, help="Max recommendations per repo")
def export_web(output, top):
    """Export compact recommendations JSON for the web frontend / extension."""
    import json
    from pathlib import Path

    src = Path(__file__).parent / "data" / "recommendations.json"
    if not src.exists():
        print("No recommendations.json found. Run 'recommend' first to generate it.")
        return

    if output is None:
        output = str(Path(__file__).parent.parent / "data")

    with open(src) as f:
        data = json.load(f)

    # Compact format: {repo: [[name, score], ...]}
    compact = {}
    for repo, recs in data.items():
        compact[repo] = [
            [r["repo"], round(r["score"], 3)]
            for r in recs[:top]
        ]

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "recommendations.json"

    with open(out_file, "w") as f:
        json.dump(compact, f, separators=(",", ":"))

    size_mb = out_file.stat().st_size / 1e6
    print(f"Exported {len(compact)} repos → {out_file} ({size_mb:.1f} MB)")


def _build_recommender(use_embeddings: bool = True):
    """Load graph and build recommender instance."""
    from graph import load_graph
    from recommender import Recommender, load_embeddings, load_repo_metadata

    repo_repo, repo_to_idx, repo_names = load_graph()
    metadata = load_repo_metadata()

    embeddings = None
    embedding_names = None
    if use_embeddings:
        try:
            embeddings, embedding_names = load_embeddings()
        except FileNotFoundError:
            pass

    return Recommender(
        repo_repo=repo_repo,
        repo_to_idx=repo_to_idx,
        repo_names=repo_names,
        repo_metadata=metadata,
        embeddings=embeddings,
        embedding_names=embedding_names,
    )


if __name__ == "__main__":
    cli()
