# main.py

import json
import os
import time
import yaml
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from searcher.search_engine import SearchEngine
from evaluation.dataset_loader import DatasetLoader

app = FastAPI(title="Semantic Search Engine")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── load search engine once at startup ──────────────────────────────────────
def get_engine():
    from searcher.search_engine import SearchEngine
    return SearchEngine("config.yaml")

engine = get_engine()


# ── load dataset queries at startup ─────────────────────────────────────────
# These are the actual queries from SciFact and NFCorpus
# We use them to show "which dataset queries matched your search"

def load_dataset_queries() -> dict:
    """
    Load all queries from SciFact and NFCorpus at startup.

    Returns:
        dict — {
            "scifact":  {query_id: query_text, ...},
            "nfcorpus": {query_id: query_text, ...},
        }
    """
    all_queries = {}

    datasets = {
        "scifact":  "data/scifact",
        "nfcorpus": "data/nfcorpus",
    }

    for name, path in datasets.items():
        if os.path.exists(path):
            try:
                loader             = DatasetLoader(path)
                all_queries[name]  = loader.load_queries()
                print(f"[Startup] Loaded {len(all_queries[name])} queries from {name}")
            except Exception as e:
                print(f"[Startup] Could not load {name} queries: {e}")
                all_queries[name] = {}
        else:
            print(f"[Startup] Dataset path not found: {path}")
            all_queries[name] = {}

    return all_queries


# load once at startup — available globally
DATASET_QUERIES = load_dataset_queries()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_eval_results() -> dict:
    path = "results/eval_all.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def extract_doc_id(filepath: str) -> str:
    if "://" in filepath:
        return filepath.split("://", 1)[1]
    return filepath


def get_dataset_from_filepath(filepath: str) -> str:
    if "scifact://"  in filepath: return "scifact"
    if "nfcorpus://" in filepath: return "nfcorpus"
    return "filesystem"


def get_file_icon(filepath: str) -> str:
    if "scifact://"  in filepath: return "🔬"
    if "nfcorpus://" in filepath: return "🏥"
    ext   = filepath.lower().split(".")[-1] if "." in filepath else ""
    icons = {
        "pdf": "📄", "docx": "📝", "txt": "📃",
        "pptx": "📊", "xlsx": "📋", "py": "🐍",
    }
    return icons.get(ext, "📄")


def find_matching_dataset_queries(
    user_query: str,
    top_results: list,
) -> list:
    """
    Find which dataset queries are semantically related to what the user typed.

    Strategy — two passes:
        1. Exact / substring match  — query text contains user words
        2. Doc-based match          — if a result doc came from dataset X,
                                      show the queries that reference that doc
                                      from the qrels (loaded separately)

    We use simple word overlap here (no extra model call needed).

    Returns:
        list of dicts — [
            {
                "query_id":   "1234",
                "query_text": "Does vitamin D cause cancer?",
                "dataset":    "scifact",
                "match_type": "text"   or "doc"
            },
            ...
        ]
    """
    matched   = []
    seen_ids  = set()

    # words from user query — lowercase, skip short words
    user_words = set(
        w.lower() for w in user_query.split()
        if len(w) > 3
    )

    # Pass 1 — text overlap match
    # check every dataset query for word overlap with user query
    for dataset_name, queries in DATASET_QUERIES.items():
        for qid, qtext in queries.items():
            q_words = set(w.lower() for w in qtext.split() if len(w) > 3)
            overlap = user_words & q_words

            # need at least 1 word overlap
            if overlap and qid not in seen_ids:
                matched.append({
                    "query_id":   qid,
                    "query_text": qtext,
                    "dataset":    dataset_name,
                    "match_type": "text",
                    "overlap":    len(overlap),
                })
                seen_ids.add(qid)

    # sort by overlap count — most overlapping queries first
    matched.sort(key=lambda x: x["overlap"], reverse=True)

    # return top 8 matched queries max
    return matched[:8]


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":          request,
        "scifact_count":    len(DATASET_QUERIES.get("scifact",  {})),
        "nfcorpus_count":   len(DATASET_QUERIES.get("nfcorpus", {})),
    })


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query:   str = Form(...),
    top_k:   int = Form(10),
    mode:    str = Form("full"),
):
    if not query.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error":   "Please enter a search query.",
        })

    t0      = time.time()
    output  = engine.search(query.strip(), top_k=top_k)
    elapsed = round(time.time() - t0, 3)

    # format search results
    results = []
    for r in output.get("results", []):
        filepath = r.get("filepath", "")
        doc_id   = extract_doc_id(filepath)
        score    = r.get("rerank_score", r.get("rrf_score", r.get("dense_score", 0)))
        snippet  = r.get("chunk_text", r.get("text", "No preview available."))

        if len(snippet) > 200:
            snippet = snippet[:200].rsplit(" ", 1)[0] + "..."

        dataset = get_dataset_from_filepath(filepath)

        results.append({
            "doc_id":   doc_id,
            "filepath": filepath,
            "score":    round(float(score), 4),
            "snippet":  snippet,
            "icon":     get_file_icon(filepath),
            "dataset":  dataset,
        })

    # find matching dataset queries
    matched_queries = find_matching_dataset_queries(query.strip(), results)

    # group matched queries by dataset for display
    matched_scifact  = [q for q in matched_queries if q["dataset"] == "scifact"]
    matched_nfcorpus = [q for q in matched_queries if q["dataset"] == "nfcorpus"]

    return templates.TemplateResponse("results.html", {
        "request":          request,
        "query":            query,
        "results":          results,
        "total":            len(results),
        "elapsed":          elapsed,
        "mode":             mode,
        "top_k":            top_k,
        "matched_scifact":  matched_scifact,
        "matched_nfcorpus": matched_nfcorpus,
        "total_matched":    len(matched_queries),
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    eval_data = load_eval_results()

    datasets = []
    for dataset_name, mode_results in eval_data.items():
        full = mode_results.get("full", {})
        datasets.append({
            "name":      dataset_name,
            "ndcg":      full.get("NDCG@10",    0.0),
            "mrr":       full.get("MRR",         0.0),
            "map":       full.get("MAP@100",     0.0),
            "recall":    full.get("Recall@100",  0.0),
            "precision": full.get("P@10",        0.0),
            "queries":   full.get("num_queries", 0),
            "modes":     mode_results,
        })

    return templates.TemplateResponse("dashboard.html", {
        "request":  request,
        "datasets": datasets,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




    # uvicorn main:app --reload --host 0.0.0.0 --port 8000