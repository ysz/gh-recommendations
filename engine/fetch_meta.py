import csv, time, httpx

TOKEN = open(".env").read().split("=", 1)[1].strip()
client = httpx.Client(
    headers={"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"},
    timeout=30,
)

repos = {}
with open("data/repos.csv") as f:
    for row in csv.DictReader(f):
        repos[row["full_name"]] = row

print(f"Fetching descriptions for {len(repos)} repos...")
start = time.time()
fetched = 0
errors = 0

for i, name in enumerate(repos):
    # Skip if already has description
    if repos[name].get("description", "").strip():
        fetched += 1
        continue

    # Check rate limit every 100 requests
    if i % 100 == 0 and i > 0:
        rl = client.get("https://api.github.com/rate_limit").json()["resources"]["core"]
        if rl["remaining"] < 10:
            wait = max(0, rl["reset"] - time.time()) + 5
            print(f"  Rate limit at {i}, waiting {wait:.0f}s...", flush=True)
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
            wait = max(0, reset - time.time()) + 5
            print(f"  Rate limit at {i}, waiting {wait:.0f}s...", flush=True)
            time.sleep(wait)
            i -= 1  # retry
        else:
            errors += 1
    except Exception:
        errors += 1

    if (i + 1) % 500 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{len(repos)} ({elapsed:.0f}s, {errors} errors)", flush=True)

with open("data/repos.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["full_name", "description", "language", "stars", "topics"])
    for r in repos.values():
        w.writerow([r["full_name"], r.get("description", ""), r.get("language", ""), r["stars"], r.get("topics", "")])

elapsed = time.time() - start
print(f"\nDone in {elapsed:.0f}s! Fetched: {fetched}, Errors: {errors}")
client.close()
