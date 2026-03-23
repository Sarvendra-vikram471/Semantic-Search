# evaluation/run_eval.py
"""
Master evaluation script.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --dataset data/scifact --mode full
    python -m evaluation.run_eval --skip-index   # reuse existing index

This script:
    1. Loads SciFact corpus, queries, qrels
    2. Indexes the corpus into your pipeline (once)
    3. Runs all 300 queries through each pipeline mode
    4. Prints an ablation table comparing dense / sparse / hybrid / full
    5. Saves results/eval_report.json
"""

import argparse
import json
import os
import time
from evaluation.dataset_loader import DatasetLoader
from evaluation.indexer_bridge import IndexerBridge
from evaluation.query_runner import QueryRunner
from evaluation.evaluator import Evaluator


MODES = ["dense", "sparse", "hybrid", "full"]

# Which metrics to show in the ablation table
DISPLAY_METRICS = ["NDCG@10", "MAP@100", "Recall@100", "P@10", "MRR"]


def print_table(results: dict):
    """Print a clean ablation table to the console."""
    col_w = 14
    header = f"{'Mode':<10}" + "".join(f"{m:>{col_w}}" for m in DISPLAY_METRICS)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for mode, metrics in results.items():
        row = f"{mode:<10}"
        for m in DISPLAY_METRICS:
            val = metrics.get(m, 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic search pipeline on SciFact")
    parser.add_argument("--dataset",    default="data/scifact", help="Path to scifact/ folder")
    parser.add_argument("--config",     default="config.yaml",  help="Path to config.yaml")
    parser.add_argument("--top-k",      default=100, type=int,  help="Results per query for eval")
    parser.add_argument("--skip-index", action="store_true",    help="Skip re-indexing (use existing index)")
    parser.add_argument("--mode",       default="all",          help="Mode: dense|sparse|hybrid|full|all")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Step 1 — load dataset
    print("\n[1/4] Loading dataset...")
    loader  = DatasetLoader(args.dataset)
    corpus  = loader.load_corpus()
    queries = loader.load_queries()
    qrels   = loader.load_qrels()

    # Step 2 — index corpus
    if not args.skip_index:
        print("\n[2/4] Indexing corpus...")
        bridge = IndexerBridge(args.config)
        bridge.index_corpus(corpus, batch_size=64)
    else:
        print("\n[2/4] Skipping indexing (--skip-index)")

    # Step 3 — run queries in each mode
    print("\n[3/4] Running queries...")
    runner    = QueryRunner(args.config)
    evaluator = Evaluator()

    modes_to_run = MODES if args.mode == "all" else [args.mode]
    all_mode_results = {}

    for mode in modes_to_run:
        print(f"\n  Mode: {mode}")
        t0 = time.time()
        ranked_results = runner.run(queries, top_k=args.top_k, mode=mode)
        elapsed = time.time() - t0

        metrics = evaluator.evaluate(ranked_results, qrels, k_values=[1, 5, 10, 100])
        metrics["query_time_s"] = round(elapsed, 2)
        all_mode_results[mode] = metrics
        print(f"  NDCG@10={metrics.get('NDCG@10', 0):.4f}  "
              f"MAP@100={metrics.get('MAP@100', 0):.4f}  "
              f"Recall@100={metrics.get('Recall@100', 0):.4f}")

    # Step 4 — report
    print("\n[4/4] Results")
    print_table(all_mode_results)

    report_path = "results/eval_report.json"
    with open(report_path, "w") as f:
        json.dump(all_mode_results, f, indent=2)
    print(f"\nSaved full report to {report_path}")


if __name__ == "__main__":
    main()