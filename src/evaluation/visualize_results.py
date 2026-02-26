"""
Evaluation Visualization for ClinicalRAG

Generates charts and tables from evaluation results for the README
and portfolio documentation. These visualizations are what make the
project look professional and demonstrate rigorous ML engineering.

Usage:
    python -m src.evaluation.visualize_results

Author: Samir Kerkar
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import numpy as np


def load_comparison_results(filepath: str) -> list[dict]:
    """Load evaluation comparison results from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_chunk_size_comparison(results: list[dict], output_dir: str = "docs"):
    """
    Generate a multi-panel chart comparing metrics across chunk sizes.
    This is the centerpiece visualization for the evaluation section.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    chunk_sizes = [r["chunk_size"] for r in results]
    keyword_recalls = [r["overall_metrics"]["avg_keyword_recall"] for r in results]
    condition_recalls = [r["overall_metrics"]["avg_condition_recall"] for r in results]
    citation_rates = [r["overall_metrics"]["citation_rate"] for r in results]
    latencies = [r["overall_metrics"]["avg_latency_seconds"] for r in results]
    diversities = [r["overall_metrics"]["avg_retrieval_diversity"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("ClinicalRAG: Impact of Chunk Size on RAG Quality", fontsize=16, fontweight="bold")
    
    # Plot 1: Keyword Recall
    axes[0, 0].bar(range(len(chunk_sizes)), keyword_recalls, color="#2563eb", alpha=0.8)
    axes[0, 0].set_xticks(range(len(chunk_sizes)))
    axes[0, 0].set_xticklabels(chunk_sizes)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Keyword Recall")
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(keyword_recalls):
        axes[0, 0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    
    # Plot 2: Condition Recall
    axes[0, 1].bar(range(len(chunk_sizes)), condition_recalls, color="#059669", alpha=0.8)
    axes[0, 1].set_xticks(range(len(chunk_sizes)))
    axes[0, 1].set_xticklabels(chunk_sizes)
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Condition Recall")
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(condition_recalls):
        axes[0, 1].text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    
    # Plot 3: Citation Rate
    axes[0, 2].bar(range(len(chunk_sizes)), citation_rates, color="#7c3aed", alpha=0.8)
    axes[0, 2].set_xticks(range(len(chunk_sizes)))
    axes[0, 2].set_xticklabels(chunk_sizes)
    axes[0, 2].set_ylabel("Rate")
    axes[0, 2].set_title("Citation Rate")
    axes[0, 2].set_ylim(0, 1)
    for i, v in enumerate(citation_rates):
        axes[0, 2].text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    
    # Plot 4: Retrieval Diversity
    axes[1, 0].bar(range(len(chunk_sizes)), diversities, color="#ea580c", alpha=0.8)
    axes[1, 0].set_xticks(range(len(chunk_sizes)))
    axes[1, 0].set_xticklabels(chunk_sizes)
    axes[1, 0].set_xlabel("Chunk Size (chars)")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Retrieval Diversity")
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(diversities):
        axes[1, 0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    
    # Plot 5: Average Latency
    axes[1, 1].bar(range(len(chunk_sizes)), latencies, color="#dc2626", alpha=0.8)
    axes[1, 1].set_xticks(range(len(chunk_sizes)))
    axes[1, 1].set_xticklabels(chunk_sizes)
    axes[1, 1].set_xlabel("Chunk Size (chars)")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].set_title("Average Latency")
    for i, v in enumerate(latencies):
        axes[1, 1].text(i, v + 0.1, f"{v:.1f}s", ha="center", fontweight="bold")
    
    # Plot 6: Combined Score (weighted average of key metrics)
    combined = [
        0.3 * kr + 0.25 * cr + 0.2 * cite + 0.15 * div + 0.1 * (1 - min(lat / 30, 1))
        for kr, cr, cite, div, lat in zip(keyword_recalls, condition_recalls, citation_rates, diversities, latencies)
    ]
    colors = ["#22c55e" if s == max(combined) else "#94a3b8" for s in combined]
    axes[1, 2].bar(range(len(chunk_sizes)), combined, color=colors, alpha=0.8)
    axes[1, 2].set_xticks(range(len(chunk_sizes)))
    axes[1, 2].set_xticklabels(chunk_sizes)
    axes[1, 2].set_xlabel("Chunk Size (chars)")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].set_title("Combined Score (weighted)")
    axes[1, 2].set_ylim(0, 1)
    for i, v in enumerate(combined):
        axes[1, 2].text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    
    plt.tight_layout()
    filepath = output_path / "chunk_size_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_category_heatmap(results: list[dict], output_dir: str = "docs"):
    """
    Generate a heatmap showing performance across categories and configs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get the latest result (or first if only one)
    result = results[-1] if isinstance(results, list) else results
    
    categories = list(result.get("category_metrics", {}).keys())
    if not categories:
        print("No category metrics to plot.")
        return
    
    metrics_to_plot = ["avg_keyword_recall", "avg_condition_recall", "citation_rate", "abstention_accuracy"]
    metric_labels = ["Keyword\nRecall", "Condition\nRecall", "Citation\nRate", "Abstention\nAccuracy"]
    
    data = []
    for cat in categories:
        row = []
        for metric in metrics_to_plot:
            val = result["category_metrics"][cat].get(metric, 0)
            row.append(val)
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in categories], fontsize=11)
    
    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(metrics_to_plot)):
            color = "white" if data[i, j] < 0.5 else "black"
            ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center", color=color, fontweight="bold")
    
    ax.set_title("ClinicalRAG: Performance by Question Category", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Score", shrink=0.8)
    
    plt.tight_layout()
    filepath = output_path / "category_performance_heatmap.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def generate_markdown_table(results: list[dict]) -> str:
    """Generate a markdown table of results for the README."""
    
    if not results:
        return "No results to display."
    
    # Check if these are chunk comparison results
    if "chunk_size" in results[0]:
        header = "| Chunk Size | Keyword Recall | Condition Recall | Citation Rate | Diversity | Latency (s) |"
        separator = "|-----------|---------------|-----------------|--------------|-----------|-------------|"
        rows = []
        for r in results:
            rows.append(
                f"| {r['chunk_size']} | "
                f"{r['overall_metrics']['avg_keyword_recall']:.1%} | "
                f"{r['overall_metrics']['avg_condition_recall']:.1%} | "
                f"{r['overall_metrics']['citation_rate']:.1%} | "
                f"{r['overall_metrics']['avg_retrieval_diversity']:.1%} | "
                f"{r['overall_metrics']['avg_latency_seconds']:.1f} |"
            )
        return "\n".join([header, separator] + rows)
    
    # Strategy comparison
    if "strategy" in results[0]:
        header = "| Strategy | Keyword Recall | Condition Recall | Citation Rate | Abstention Acc | Latency (s) |"
        separator = "|----------|---------------|-----------------|--------------|----------------|-------------|"
        rows = []
        for r in results:
            label = r["strategy"]["label"]
            rows.append(
                f"| {label} | "
                f"{r['overall_metrics']['avg_keyword_recall']:.1%} | "
                f"{r['overall_metrics']['avg_condition_recall']:.1%} | "
                f"{r['overall_metrics']['citation_rate']:.1%} | "
                f"{r['overall_metrics']['abstention_accuracy']:.1%} | "
                f"{r['overall_metrics']['avg_latency_seconds']:.1f} |"
            )
        return "\n".join([header, separator] + rows)
    
    return "Unknown result format."


if __name__ == "__main__":
    eval_dir = Path("data/evaluation_results")
    
    # Try to load and visualize chunk comparison
    chunk_file = eval_dir / "chunk_size_comparison.json"
    if chunk_file.exists():
        print("Generating chunk size comparison charts...")
        results = load_comparison_results(str(chunk_file))
        plot_chunk_size_comparison(results)
        print("\nMarkdown table for README:")
        print(generate_markdown_table(results))
    
    # Try to load and visualize strategy comparison
    strategy_file = eval_dir / "retrieval_strategy_comparison.json"
    if strategy_file.exists():
        print("\nGenerating retrieval strategy charts...")
        results = load_comparison_results(str(strategy_file))
        print("\nMarkdown table for README:")
        print(generate_markdown_table(results))
    
    # Generate category heatmap from most recent evaluation
    eval_files = sorted(eval_dir.glob("eval_*.json"))
    if eval_files:
        print("\nGenerating category heatmap...")
        with open(eval_files[-1], "r") as f:
            data = json.load(f)
        plot_category_heatmap(data["summary"])
    
    if not chunk_file.exists() and not strategy_file.exists():
        print("No evaluation results found. Run evaluations first:")
        print("  python -m src.evaluation.evaluate_rag --mode quick")
        print("  python -m src.evaluation.evaluate_rag --mode full")
        print("  python -m src.evaluation.evaluate_rag --mode chunk_comparison")
