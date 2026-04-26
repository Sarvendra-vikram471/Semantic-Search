import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def load_json(filename):
    with (RESULTS_DIR / filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_dataset_comparison():
    data = load_json("eval_all.json")
    metrics = ["NDCG@10", "MAP@100", "Recall@100", "MRR", "P@10"]
    datasets = list(data.keys())

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 5))

    for i, dataset in enumerate(datasets):
        values = [data[dataset]["full"].get(metric, 0) for metric in metrics]
        positions = [pos + (i - 0.5) * width for pos in x]
        plt.bar(positions, values, width=width, label=dataset.capitalize())

    plt.title("Dataset Comparison - Full Search Model")
    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Score")
    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dataset_comparison.png", dpi=300)
    plt.show()


def plot_model_comparison():
    data = load_json("eval_report.json")
    metrics = ["NDCG@10", "MAP@100", "Recall@100", "MRR", "P@10"]
    models = list(data.keys())

    x = range(len(metrics))
    width = 0.18

    plt.figure(figsize=(11, 5))

    offset_start = -(len(models) - 1) / 2
    for i, model in enumerate(models):
        values = [data[model].get(metric, 0) for metric in metrics]
        positions = [pos + (offset_start + i) * width for pos in x]
        plt.bar(positions, values, width=width, label=model.capitalize())

    plt.title("Model / Retrieval Mode Comparison - SciFact")
    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Score")
    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_dataset_comparison()
    plot_model_comparison()
