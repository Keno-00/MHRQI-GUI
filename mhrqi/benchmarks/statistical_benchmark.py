"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Statistical Treatment: Wilcoxon Signed-Rank Test                            ║
║  Benchmark MHRQI vs State-of-the-Art on Medical OCT Images                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Folder Structure:
benchmark/
└── YYYYMMDD_HHMMSS/
    ├── cnv1/
    │   ├── original.png
    │   ├── denoised_bm3d.png
    │   ├── denoised_nlmeans.png
    │   ├── denoised_srad.png
    │   ├── denoised_proposed.png
    │   └── report_*.png
    ├── cnv2/
    │   └── ...
    └── metrics/
        ├── raw_results.json
        ├── summary.json
        ├── speckle_consistency_metrics.png
        └── no-reference_quality.png

SOTA Reference: BM3D (for Full-Reference metrics)
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
from scipy import stats

from mhrqi.benchmarks import compare_to
from mhrqi.cli import main
from mhrqi.utils import visualization as plots

# Medical image paths. Only use DR images for this case. Never use Images in Folders
MEDICAL_IMAGES = [
    "resources/cnv1.jpeg",  # 8 CNV
    "resources/cnv2.jpeg",
    "resources/cnv3.jpeg",
    "resources/cnv4.jpeg",
    "resources/cnv5.jpeg",
    "resources/cnv6.jpeg",
    "resources/cnv7.jpeg",
    "resources/cnv8.jpeg",
    "resources/dme1.jpeg",  # 8 DME
    "resources/dme2.jpeg",
    "resources/dme3.jpeg",
    "resources/dme4.jpeg",
    "resources/dme5.jpeg",
    "resources/dme6.jpeg",
    "resources/dme7.jpeg",
    "resources/dme8.jpeg",
    "resources/drusen1.jpeg",  # 8 Drusen
    "resources/drusen2.jpeg",
    "resources/drusen3.jpeg",
    "resources/drusen4.jpeg",
    "resources/drusen5.jpeg",
    "resources/drusen6.jpeg",
    "resources/drusen7.jpeg",
    "resources/drusen8.jpeg",
    "resources/normal1.jpeg",  # 8 Normal
    "resources/normal2.jpeg",
    "resources/normal3.jpeg",
    "resources/normal4.jpeg",
    "resources/normal5.jpeg",
    "resources/normal6.jpeg",
    "resources/normal7.jpeg",
    "resources/normal8.jpeg",
]

# total 32 images

# Metrics for comparison (No synthetic clean reference available)
SPECKLE_METRICS_LOWER = ["SSI", "SMPI"]  # Lower is better
SPECKLE_METRICS_HIGHER = ["ENL", "CNR", "NSF"]  # Higher is better
STRUCTURAL_METRICS = ["EPF", "EPI", "OMQDI"]  # Higher is better
# NIQE computed but not reported (biased for medical images)
ALL_SPECKLE_METRICS = SPECKLE_METRICS_LOWER + SPECKLE_METRICS_HIGHER

# Methods to compare
METHODS = ["bm3d", "nlmeans", "srad", "proposed"]


def run_benchmark(n=64, _strength=None):
    """
    Run benchmark on all medical images and collect metrics.

    Folder structure:
    - benchmark/timestamp/imagename/ for each image
    - benchmark/timestamp/metrics/ for aggregated stats

    Args:
        n: Image size
        strength: Denoiser strength parameter

    Returns:
        results: Dict mapping image -> method -> metrics
        base_dir: Path to benchmark output directory
    """
    # Create base directory: benchmark/timestamp/ (matching runs/ format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Same as runs/ folder
    base_dir = os.path.join("benchmark", timestamp)
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    all_results = {}

    for img_path in MEDICAL_IMAGES:
        img_name = os.path.basename(img_path).replace(".jpeg", "")
        img_dir = os.path.join(base_dir, img_name)
        os.makedirs(img_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing: {img_name}")
        print(f"Output: {img_dir}")
        print(f"{'=' * 60}")

        # Run MHRQI pipeline
        orig, recon, run_dir = main.main(
            shots=1000,
            n=n,
            d=2,
            denoise=True,
            use_shots=False,
            fast=True,
            verbose_plots=False,
            img_path=img_path,
            run_comparison=False,
        )

        # Save original
        cv2.imwrite(os.path.join(img_dir, "original.png"), orig)

        # Run comparison with BM3D as SOTA reference for FR metrics
        noisy_img = compare_to.to_float01(orig)

        # Run comparison (no synthetic clean reference)
        comparison_results = compare_to.compare_to(
            noisy_img,
            proposed_img=compare_to.to_float01(recon),
            methods="all",
            plot=True,  # Generate report plots
            save=True,
            save_prefix="denoised",
            save_dir=img_dir,
            reference_image=None,  # No synthetic reference
        )

        # Extract metrics for each method
        img_results = {}
        for r in comparison_results:
            method_name = r["name"]
            img_results[method_name] = r["metrics"]

        all_results[img_name] = img_results

    # Save raw results to metrics folder
    with open(os.path.join(metrics_dir, "raw_results.json"), "w") as f:
        serializable = {}
        for img, methods in all_results.items():
            serializable[img] = {}
            for method, metrics in methods.items():
                serializable[img][method] = {
                    k: float(v) if not np.isnan(v) else None for k, v in metrics.items()
                }
        json.dump(serializable, f, indent=2)

    return all_results, base_dir, metrics_dir


def wilcoxon_test(all_results, method1, method2, metric):
    """
    Perform Wilcoxon signed-rank test between two methods on a specific metric.
    """
    diffs = []
    for _img_name, methods in all_results.items():
        if method1 in methods and method2 in methods:
            v1 = methods[method1].get(metric, float("nan"))
            v2 = methods[method2].get(metric, float("nan"))
            if not np.isnan(v1) and not np.isnan(v2):
                diffs.append(v1 - v2)

    if len(diffs) < 3:
        return None, None, diffs

    try:
        stat, p_value = stats.wilcoxon(diffs)
        return stat, p_value, diffs
    except Exception as e:
        print(f"Wilcoxon test failed: {e}")
        return None, None, diffs


def create_results_table(all_results, metrics_dir):
    """
    Create summary tables for all metrics.
    """
    # Aggregate metrics across all images
    all_metrics_list = ALL_SPECKLE_METRICS + STRUCTURAL_METRICS
    method_metrics = {m: {k: [] for k in all_metrics_list} for m in METHODS}

    for _img_name, methods in all_results.items():
        for method in METHODS:
            if method in methods:
                for metric in all_metrics_list:
                    val = methods[method].get(metric, float("nan"))
                    if not np.isnan(val):
                        method_metrics[method][metric].append(val)

    # Calculate means and stds
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in all_metrics_list:
            vals = method_metrics[method][metric]
            if vals:
                summary[method][metric] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "n": len(vals),
                }
            else:
                summary[method][metric] = {"mean": float("nan"), "std": float("nan"), "n": 0}

    # Print table
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS (Mean ± Std across all images)")
    print("=" * 100)

    all_metrics = all_metrics_list
    header = f"{'Method':<12}" + "".join(f"{m:<15}" for m in all_metrics)
    print(header)
    print("-" * 80)

    for method in METHODS:
        row = f"{method:<12}"
        for metric in all_metrics:
            s = summary[method][metric]
            if s["n"] > 0:
                row += f"{s['mean']:.4f}±{s['std']:.4f}  "
            else:
                row += "N/A           "
        print(row)

    # ==========================================================================
    # STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test)
    # All metrics are paired samples (same images, different methods)
    # ==========================================================================

    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test)")
    print("=" * 90)

    # Define metric categories with hypotheses
    metric_categories = [
        {
            "name": "Speckle Reduction (Lower Better)",
            "metrics": SPECKLE_METRICS_LOWER,
            "higher_better": False,  # Lower is better for SSI, SMPI
            "H0": "MHRQI achieves the same speckle reduction as [competitor]",
            "H1": "MHRQI achieves different speckle reduction than [competitor]",
        },
        {
            "name": "Speckle Reduction (Higher Better)",
            "metrics": SPECKLE_METRICS_HIGHER,
            "higher_better": True,  # Higher is better for ENL, CNR, NSF
            "H0": "MHRQI achieves the same speckle reduction as [competitor]",
            "H1": "MHRQI achieves different speckle reduction than [competitor]",
        },
        {
            "name": "Structural Similarity",
            "metrics": STRUCTURAL_METRICS,
            "higher_better": True,  # Higher is better for EPF, EPI, OMQDI
            "H0": "MHRQI achieves the same structural preservation as [competitor]",
            "H1": "MHRQI achieves different structural preservation than [competitor]",
        },
        # Naturalness removed (NIQE computed but not reported - biased for medical images)
    ]

    # Store results for summary
    stat_results = []

    for category in metric_categories:
        print(f"\n{'─' * 90}")
        print(f"Category: {category['name']}")
        print(f"  H₀: {category['H0']}")
        print(f"  H₁: {category['H1']}")
        print(f"  Direction: {'Higher' if category['higher_better'] else 'Lower'} is better")
        print(f"{'─' * 90}")

        for other_method in ["bm3d", "nlmeans", "srad"]:
            print(f"\n  {other_method.upper()} vs MHRQI:")

            for metric in category["metrics"]:
                stat, p_val, diffs = wilcoxon_test(all_results, "proposed", other_method, metric)

                if p_val is not None:
                    mean_diff = np.mean(diffs)

                    # Interpret based on direction
                    if category["higher_better"]:
                        mhrqi_better = mean_diff > 0
                    else:
                        mhrqi_better = mean_diff < 0  # Lower is better

                    # Determine significance and language
                    if p_val < 0.05:
                        decision = "Reject H₀"
                        if mhrqi_better:
                            interpretation = "MHRQI significantly better"
                        else:
                            interpretation = f"{other_method} significantly better"
                    else:
                        decision = "Fail to reject H₀"
                        interpretation = "Comparable (no significant difference)"

                    sig = (
                        "***"
                        if p_val < 0.001
                        else "**"
                        if p_val < 0.01
                        else "*"
                        if p_val < 0.05
                        else "n.s."
                    )

                    print(f"    {metric:<10}: p={p_val:.4f} ({sig:<4}) → {decision}")
                    print(f"               Mean Δ={mean_diff:+.4f} → {interpretation}")

                    stat_results.append(
                        {
                            "category": category["name"],
                            "competitor": other_method,
                            "metric": metric,
                            "p_value": p_val,
                            "mean_diff": mean_diff,
                            "interpretation": interpretation,
                            "significant": p_val < 0.05,
                        }
                    )
                else:
                    print(f"    {metric:<10}: insufficient data (need ≥3 paired samples)")

    # Save statistical results
    with open(os.path.join(metrics_dir, "statistical_results.json"), "w") as f:
        json.dump(
            stat_results,
            f,
            indent=2,
            default=lambda x: int(x)
            if isinstance(x, (bool, np.bool_))
            else float(x)
            if isinstance(x, np.floating)
            else x,
        )

    # Save summary
    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(
            summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x
        )

    return summary, stat_results


def create_visualization(all_results, metrics_dir):
    """
    Create separate visualizations for each metric category.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = METHODS

    metric_groups = [
        ("Speckle Reduction (Lower Better)", SPECKLE_METRICS_LOWER, False),
        ("Speckle Reduction (Higher Better)", SPECKLE_METRICS_HIGHER, True),
        ("Structural Similarity Metrics", STRUCTURAL_METRICS, True),
        # Naturalness removed (NIQE computed but not reported)
    ]

    for group_name, metrics, higher_better in metric_groups:
        method_means = {m: [] for m in methods}
        method_stds = {m: [] for m in methods}

        for metric in metrics:
            for method in methods:
                vals = []
                for img_results in all_results.values():
                    if method in img_results:
                        v = img_results[method].get(metric, float("nan"))
                        if not np.isnan(v):
                            vals.append(v)
                if vals:
                    # Clip to 10000 for visualization if needed
                    m_val = np.mean(vals)
                    s_val = np.std(vals)
                    if m_val > 10000:
                        m_val = 10000
                    if s_val > 10000:
                        s_val = 10000
                    method_means[method].append(m_val)
                    method_stds[method].append(s_val)
                else:
                    method_means[method].append(0)
                    method_stds[method].append(0)

        # Skip if no data
        if all(all(v == 0 for v in method_means[m]) for m in methods):
            continue

        width = 0.4

        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)), sharex=False)
        if len(metrics) == 1:
            axes = [axes]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        labels = ["BM3D", "NL-Means", "SRAD", "MHRQI (Ours)"]

        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            means = [method_means[m][idx] for m in methods]
            stds = [method_stds[m][idx] for m in methods]

            # Use bars instead of grouped bars since we have subplots
            ax.bar(methods, means, width, yerr=stds, capsize=5, color=colors)

            ax.set_title(f"Metric: {metric}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Score", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            # Use labels for methods
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(labels)

            direction_note = "↑ Higher is better" if higher_better else "↓ Lower is better"
            ax.annotate(
                direction_note,
                xy=(0.98, 0.02),
                xycoords="axes fraction",
                ha="right",
                fontsize=9,
                style="italic",
                color="gray",
            )

        fig.suptitle(group_name, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = (
            group_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "")
            + ".png"
        )
        plt.savefig(os.path.join(metrics_dir, filename), dpi=150)
        plt.close()

        print(f"  Saved: {filename}")

    print(f"\nVisualizations saved to: {metrics_dir}")


def create_summary_heatmap(stat_results, metrics_dir):
    """
    Create a WIN/TIE/LOSS heatmap from pre-computed statistical results.

    Args:
        stat_results: List of dicts from create_results_table() (same structure
                      as statistical_results.json).
        metrics_dir:  Directory where the PNG is saved.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not stat_results:
        print("  No statistical results to plot; skipping heatmap.")
        return

    competitors = sorted({d["competitor"] for d in stat_results})
    metrics = sorted({d["metric"] for d in stat_results})

    # Preserve category grouping
    categories = {d["metric"]: d["category"] for d in stat_results}
    metrics.sort(key=lambda x: categories[x])

    # Build matrix: 2 = WIN, 1 = TIE, 0 = LOSS
    heatmap_data = np.zeros((len(competitors), len(metrics)))
    for d in stat_results:
        row = competitors.index(d["competitor"])
        col = metrics.index(d["metric"])
        interp = d["interpretation"].lower()
        if "mhrqi significantly better" in interp:
            heatmap_data[row, col] = 2
        elif "comparable" in interp:
            heatmap_data[row, col] = 1
        # else 0 (competitor better)

    fig, ax = plt.subplots(figsize=(14, 7))

    cmap = mcolors.ListedColormap(["#FF8A8A", "#FFFBD1", "#A3EBB1"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect="auto")

    # Grid lines
    ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(competitors) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Axis labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(competitors)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold", rotation=0)
    ax.set_yticklabels([c.upper() for c in competitors], fontsize=11, fontweight="bold")

    # Cell annotations
    for i in range(len(competitors)):
        for j in range(len(metrics)):
            val = heatmap_data[i, j]
            label = "WIN" if val == 2 else "TIE" if val == 1 else "LOSS"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="#333333",
                fontsize=11,
                fontweight="bold",
            )

    # Category span headers
    unique_cats, cat_starts, current_cat = [], [], None
    for i, m in enumerate(metrics):
        if categories[m] != current_cat:
            unique_cats.append(categories[m])
            cat_starts.append(i)
            current_cat = categories[m]

    for i, cat in enumerate(unique_cats):
        start = cat_starts[i]
        end = cat_starts[i + 1] if i + 1 < len(cat_starts) else len(metrics)
        mid = (start + end - 1) / 2
        ax.annotate(
            "",
            xy=(start - 0.4, -0.6),
            xytext=(end - 0.6, -0.6),
            xycoords="data",
            textcoords="data",
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 1.5},
        )
        ax.text(
            mid,
            -0.8,
            cat,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            style="italic",
            color="#555555",
        )

    ax.spines[:].set_visible(False)
    ax.set_title(
        "MHRQI Performance Benchmark Summary\n(Statistical Significance vs SOTA)",
        fontsize=15,
        fontweight="bold",
        pad=50,
    )

    legend_elements = [
        Patch(facecolor="#A3EBB1", label="MHRQI Significantly Better (p < 0.05)"),
        Patch(facecolor="#FFFBD1", label="Competitive / No Significant Difference"),
        Patch(facecolor="#FF8A8A", label="Competitor Significantly Better (p < 0.05)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    output_path = os.path.join(metrics_dir, "benchmark_summary_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: benchmark_summary_heatmap.png")


if __name__ == "__main__":
    print("=" * 60)
    print("MHRQI Statistical Benchmark")
    print("(No synthetic clean reference - degraded reference only)")
    print("=" * 60)

    # Run benchmark
    all_results, base_dir, metrics_dir = run_benchmark(n=16, strength=1.65)

    # Create results table and statistical tests
    summary, stat_results = create_results_table(all_results, metrics_dir)

    # Create visualizations
    create_visualization(all_results, metrics_dir)
    create_summary_heatmap(stat_results, metrics_dir)

    print(f"\n{'=' * 60}")
    print(f"All results saved to: {base_dir}")
    print(f"Metrics and charts: {metrics_dir}")
    print("=" * 60)
