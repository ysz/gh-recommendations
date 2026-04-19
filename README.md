# GitHub Repository Recommendations

A Chrome extension that shows similar repository recommendations directly in the GitHub sidebar.

Recommendations are based on how people star repos: if users who starred repo A also tend to star repo B, those repos are likely related.

<img width="800" height="483" alt="Image" src="https://github.com/user-attachments/assets/de622e71-a5ef-42f8-9851-581ac4eba92b" />

## Install the extension

1. Clone this repo:
   ```bash
   git clone https://github.com/ysz/gh-recommendations.git
   ```
2. Open `chrome://extensions` in Chrome
3. Enable **Developer mode** (top right)
4. Click **Load unpacked** and select the cloned folder
5. Open any repository on GitHub -- a "Similar Repositories" panel appears at the bottom of the sidebar

Star counts and descriptions are loaded live from GitHub.

Currently covers repositories with 100+ stars (~9K repos in the dataset).

## How recommendations work

Three signals are combined to score each recommendation:

1. **Co-star similarity (40%)** -- collaborative filtering on star data. Two repos are similar if many of the same users starred both.
2. **Description embeddings (35%)** -- semantic similarity of repo descriptions using [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (1024-dim vectors).
3. **Topic overlap (15%)** -- Jaccard similarity between GitHub topic tags.

Star data comes from [GH Archive](https://www.gharchive.org/) via Google BigQuery. Repo metadata comes from the [GitHub REST API](https://docs.github.com/en/rest).

## Project structure

```
manifest.json            # Chrome extension manifest
content.js               # Injects recommendations into GitHub sidebar
styles.css               # Minimal styling overrides
icons/                   # Extension icons
data/
  recommendations.json   # Pre-computed recommendations (~3 MB)
engine/                  # Recommendation pipeline (Python)
  main.py                # CLI: collect, build, recommend, evaluate
  collector.py           # Data collection (GitHub API + BigQuery)
  graph.py               # Sparse matrix construction
  recommender.py         # Hybrid recommendation engine
  evaluate.py            # Held-out evaluation (Hit Rate, MRR)
```

## Rebuilding recommendations

The `engine/` folder contains the Python pipeline that generates `data/recommendations.json`. You only need this if you want to rebuild the data yourself.

```bash
cd engine
uv sync        # or: pip install -e .

# Get star data from BigQuery (prints SQL to run in console.cloud.google.com)
python main.py bigquery-sql

# Load exported CSV
python main.py load-bigquery stars.csv

# Build graph + embeddings + export
python main.py build
python main.py embeddings
python main.py export-web
```

Requires Python 3.12+.

## Data sources and attribution

- **[GH Archive](https://www.gharchive.org/)** -- public GitHub event data, available on [Google BigQuery](https://console.cloud.google.com/bigquery).
- **[GitHub REST API](https://docs.github.com/en/rest)** -- repository metadata. Subject to [GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service).
- **[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** -- embedding model by Alibaba/Qwen (Apache 2.0).

## License

MIT
