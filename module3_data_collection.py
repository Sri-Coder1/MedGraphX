import requests
import feedparser
import urllib.parse


# -----------------------------
# WIKIPEDIA DATA COLLECTION
# -----------------------------
def fetch_wikipedia_data(query):
    if not query or not str(query).strip():
        return {"error": "Empty query"}

    headers = {"User-Agent": "MedGraphX/1.0 (https://example.com)"}

    def _summary_for_title(title):
        title_enc = urllib.parse.quote(title)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_enc}"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None, resp
        try:
            return resp.json(), resp
        except ValueError:
            return None, resp

    # First try direct title lookup
    try:
        data, resp = _summary_for_title(query)
        if data:
            return {
                "source": "wikipedia",
                "entity": data.get("title", query),
                "description": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
            }

        # If not found (404) or invalid JSON, fall back to search API
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        sresp = requests.get(search_url, params=params, headers=headers, timeout=10)
        sresp.raise_for_status()
        sdata = sresp.json()
        hits = sdata.get("query", {}).get("search", [])
        if not hits:
            return {"source": "wikipedia", "query": query, "entity": None, "description": "", "url": ""}

        best_title = hits[0]["title"]
        data2, resp2 = _summary_for_title(best_title)
        if data2:
            return {
                "source": "wikipedia",
                "entity": data2.get("title", best_title),
                "description": data2.get("extract", ""),
                "url": data2.get("content_urls", {}).get("desktop", {}).get("page", "")
            }

        # If summary fetch still failed, return helpful error
        return {"error": f"Could not fetch summary (status {resp2.status_code if resp2 is not None else 'N/A'})", "raw": resp2.text if resp2 is not None else None}

    except requests.RequestException as e:
        return {"error": str(e)}


# -----------------------------
# ARXIV DATA COLLECTION
# -----------------------------
def fetch_arxiv_data(query):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"

    try:
        feed = feedparser.parse(url)

        papers = []

        for entry in feed.entries:
            papers.append({
                "title": entry.title,
                "summary": entry.summary,
                "published": entry.published,
                "link": entry.link
            })

        return {
            "source": "arxiv",
            "query": query,
            "papers": papers
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# PUBMED DATA COLLECTION
# -----------------------------
def fetch_pubmed_data(query):
    try:
        # Step 1: Search articles
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": 5,
            "retmode": "json"
        }

        search_resp = requests.get(search_url, params=search_params)
        search_data = search_resp.json()

        ids = search_data.get("esearchresult", {}).get("idlist", [])

        if not ids:
            return {"source": "pubmed", "articles": []}

        # Step 2: Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }

        fetch_resp = requests.get(fetch_url, params=fetch_params)
        fetch_data = fetch_resp.json().get("result", {})

        articles = []

        for pid in ids:
            article = fetch_data.get(pid, {})
            if not article:
                continue

            authors = [a.get("name", "") for a in article.get("authors", [])]

            articles.append({
                "title": article.get("title", ""),
                "authors": authors[:3],
                "journal": article.get("source", ""),
                "published": article.get("pubdate", ""),
                "link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            })

        return {
            "source": "pubmed",
            "query": query,
            "articles": articles
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# UNIFIED DATA COLLECTION
# -----------------------------
def collect_data(source, query):

    source = source.lower()

    if source == "wikipedia":
        return fetch_wikipedia_data(query)

    elif source == "arxiv":
        return fetch_arxiv_data(query)

    elif source == "pubmed":
        return fetch_pubmed_data(query)

    else:
        return {"error": "Invalid source"}