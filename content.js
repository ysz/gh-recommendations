(() => {
  // Load from extension's own bundled file by default
  const BUNDLED_DATA_URL = chrome.runtime.getURL("data/recommendations.json");

  let recsCache = null;
  let repoInfoCache = {};
  let currentRepo = null;

  function getRepoFromURL() {
    const match = location.pathname.match(/^\/([^/]+\/[^/]+)\/?$/);
    if (!match) return null;
    const excluded = [
      "settings",
      "notifications",
      "sponsors",
      "organizations",
      "orgs",
    ];
    const owner = match[1].split("/")[0];
    if (excluded.includes(owner)) return null;
    return match[1];
  }

  function isRepoPage() {
    return (
      document.querySelector('[data-pjax="#repo-content-pjax-container"]') !==
        null ||
      document.querySelector(".repository-content") !== null ||
      document.querySelector("#repository-container-header") !== null
    );
  }

  async function loadRecommendations() {
    if (recsCache) return recsCache;

    const url = BUNDLED_DATA_URL;
    try {
      const resp = await fetch(url);
      if (!resp.ok) return null;
      recsCache = await resp.json();
      return recsCache;
    } catch {
      return null;
    }
  }

  async function fetchRepoInfo(repoName) {
    if (repoInfoCache[repoName]) return repoInfoCache[repoName];

    try {
      const resp = await fetch(`https://github.com/${repoName}`, {
        headers: { Accept: "text/html" },
      });
      if (!resp.ok) return null;
      const html = await resp.text();
      const doc = new DOMParser().parseFromString(html, "text/html");

      let stars = null;
      const starBtn = doc.querySelector("#repo-stars-counter-star");
      if (starBtn) {
        const raw = starBtn.getAttribute("title") || starBtn.textContent;
        stars = parseInt(raw.replace(/[^0-9]/g, ""), 10) || null;
      }

      let description = "";
      // Try meta description: "Description text - owner/repo"
      const metaDesc = doc.querySelector('meta[name="description"]');
      if (metaDesc) {
        const content = metaDesc.getAttribute("content") || "";
        const match = content.match(/^(.+?)\s*-\s*\S+\/\S+\s*$/);
        if (match && !match[1].startsWith("Contribute to")) {
          description = match[1].trim();
        }
      }
      // Fallback: title tag "GitHub - owner/repo: Description · GitHub"
      if (!description) {
        const titleEl = doc.querySelector("title");
        if (titleEl) {
          const match = titleEl.textContent.match(
            /^GitHub - [^:]+:\s*(.+?)(?:\s*·\s*GitHub)?$/
          );
          if (match) {
            description = match[1];
          }
        }
      }

      const info = { stars, description };
      repoInfoCache[repoName] = info;
      return info;
    } catch {
      return null;
    }
  }

  function formatStars(n) {
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return String(n);
  }

  function createSidebarRow(content) {
    // Match GitHub's BorderGrid-row structure
    const row = document.createElement("div");
    row.className = "BorderGrid-row";
    row.id = "gh-recs-row";

    const cell = document.createElement("div");
    cell.className = "BorderGrid-cell";
    cell.appendChild(content);

    row.appendChild(cell);
    return row;
  }

  function createPanel(recs) {
    const container = document.createElement("div");

    const title = document.createElement("h2");
    title.className = "h4 mb-3";
    title.textContent = "Similar Repositories";
    container.appendChild(title);

    if (!recs || recs.length === 0) {
      const empty = document.createElement("p");
      empty.className = "color-fg-muted f6";
      empty.textContent = "No recommendations available for this repo";
      container.appendChild(empty);
      return createSidebarRow(container);
    }

    const list = document.createElement("ul");
    list.className = "list-style-none";
    list.style.cssText = "padding: 0; margin: 0;";

    const top = recs.slice(0, 10);
    for (const rec of top) {
      const repoName = rec[0];
      const score = rec[1];

      const li = document.createElement("li");
      li.className = "py-2";
      li.style.cssText = "border-bottom: 1px solid var(--borderColor-muted, #d8dee4);";

      const topRow = document.createElement("div");
      topRow.className = "d-flex justify-content-between align-items-center";

      const link = document.createElement("a");
      link.href = `/${repoName}`;
      link.className = "Link--primary f6 text-bold";
      link.style.cssText =
        "overflow: hidden; text-overflow: ellipsis; white-space: nowrap;";
      link.textContent = repoName;

      const meta = document.createElement("span");
      meta.className = "color-fg-muted f6 no-wrap ml-2";
      meta.textContent = `${Math.round(score * 100)}%`;

      topRow.appendChild(link);
      topRow.appendChild(meta);
      li.appendChild(topRow);

      const desc = document.createElement("p");
      desc.className = "color-fg-muted f6 mb-0 mt-1";
      desc.style.cssText = "display: none;";
      li.appendChild(desc);

      fetchRepoInfo(repoName).then((info) => {
        if (info) {
          if (info.stars != null) {
            meta.textContent = `★ ${formatStars(info.stars)}`;
          }
          if (info.description) {
            desc.textContent = info.description;
            desc.style.display = "block";
          }
        }
      });

      list.appendChild(li);
    }

    container.appendChild(list);
    return createSidebarRow(container);
  }

  function findSidebar() {
    const selectors = [
      ".BorderGrid.BorderGrid--spacious",
      ".repository-content .BorderGrid",
      "[data-testid='repo-sidebar']",
    ];
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return null;
  }

  async function inject() {
    const repoName = getRepoFromURL();
    if (!repoName || repoName === currentRepo) return;
    if (!isRepoPage()) return;

    currentRepo = repoName;

    const old = document.getElementById("gh-recs-row");
    if (old) old.remove();

    const sidebar = findSidebar();
    if (!sidebar) return;

    // Loading state
    const loadingContent = document.createElement("div");
    const loadingTitle = document.createElement("h2");
    loadingTitle.className = "h4 mb-3";
    loadingTitle.textContent = "Similar Repositories";
    loadingContent.appendChild(loadingTitle);
    const loadingText = document.createElement("p");
    loadingText.className = "color-fg-muted f6";
    loadingText.textContent = "Loading...";
    loadingContent.appendChild(loadingText);
    const loadingRow = createSidebarRow(loadingContent);
    sidebar.appendChild(loadingRow);

    const data = await loadRecommendations();
    const recs = data ? data[repoName] : null;

    const panel = createPanel(recs);
    loadingRow.replaceWith(panel);
  }

  inject();

  document.addEventListener("turbo:render", () => {
    currentRepo = null;
    inject();
  });

  let lastURL = location.href;
  new MutationObserver(() => {
    if (location.href !== lastURL) {
      lastURL = location.href;
      currentRepo = null;
      inject();
    }
  }).observe(document.body, { childList: true, subtree: true });
})();
