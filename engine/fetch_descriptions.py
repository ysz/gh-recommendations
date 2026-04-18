"""Fetch repo descriptions from GitHub API. Runs at 60 req/hour."""
import csv, time, os, json
import httpx

TOKEN = os.environ.get("GITHUB_TOKEN") or open(".env").read().split("=",1)[1].strip()
DATA = "data"

client = httpx.Client(
    headers={"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"},
    timeout=30,
)

# Load repo list
repos = {}
with open(f"{DATA}/repos.csv") as f:
    for row in csv.DictReader(f):
        repos[row["full_name"]] = row

# Check which already have descriptions
need_fetch = [name for name, r in repos.items() if not r.get("description")]
print(f"Need descriptions for {len(need_fetch)} repos")

fetched = 0
for i, name in enumerate(need_fetch):
    # Rate limit check
    if fetched > 0 and fetched % 55 == 0:
        r = client.get("https://api.github.com/rate_limit")
        remaining = r.json()["resources"]["core"]["remaining"]
        if remaining < 5:
            reset = r.json()["resources"]["core"]["reset"]
            wait = max(0, reset - time.time()) + 2
            print(f"  Rate limit, waiting {wait:.0f}s...")
            time.sleep(wait)

    try:
        r = client.get(f"https://api.github.com/repos/{name}")
        if r.status_code == 200:
            d = r.json()
            repos[name]["description"] = (d.get("description") or "")[:500]
            repos[name]["language"] = d.get("language") or ""
            repos[name]["topics"] = ",".join(d.get("topics", []))
            fetched += 1
        elif r.status_code == 403:
            reset = int(r.headers.get("x-ratelimit-reset", 0))
            wait = max(0, reset - time.time()) + 2
            print(f"  [{i}] Rate limited, waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
        else:
            print(f"  [{i}] {name}: {r.status_code}")
    except Exception as e:
        print(f"  [{i}] {name}: {e}")

    if fetched % 50 == 0 and fetched > 0:
        print(f"  Fetched {fetched}/{len(need_fetch)}...")
        # Save progress
        with open(f"{DATA}/repos.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["full_name", "description", "language", "stars", "topics"])
            for r in repos.values():
                w.writerow([r["full_name"], r.get("description",""), r.get("language",""), r["stars"], r.get("topics","")])

# Final save
with open(f"{DATA}/repos.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["full_name", "description", "language", "stars", "topics"])
    for r in repos.values():
        w.writerow([r["full_name"], r.get("description",""), r.get("language",""), r["stars"], r.get("topics","")])

print(f"\nDone! Fetched {fetched} descriptions.")
