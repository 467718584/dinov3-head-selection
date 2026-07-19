"""Regenerate Fig 8 (single-prototype) with fixed legend positioning.

Fix v3 (2026-07-26): R6 reviewer feedback
- Issue: green annotation "All anomalies are far from centroid" in panel (b)
  was occluded by the legend box + overlapping with the title.
- Fix: relocate legend to `loc='lower right'`; move green annotation to
  `ax.text(x, y, ..., bbox=dict(facecolor='white', alpha=0.85, edgecolor='green'))`
  with explicit xy coordinates outside the dense data cluster.

Follows project Python style (PEP 8, type annotations, logging, immutable
data via numpy arrays).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("regen_fig8_v3")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,        # smaller legend font
    "lines.linewidth": 1.5,
    "lines.markersize": 7,
    "figure.dpi": 200,
})

OUT_DIR = Path("D:/project/docker-images/research/01_direction_visual_perception/01_research/papers/paper1_TIM/versions/v28.4.2_camera_ready/figures")


def _generate_data(seed: int, n_normal: int = 100, n_anomaly: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for Fig 8 (left + right panels share base data)."""
    rng = np.random.default_rng(seed)
    normal: np.ndarray = rng.standard_normal((n_normal, 2))
    anomaly: np.ndarray = rng.standard_normal((n_anomaly, 2)) + 3
    memory: np.ndarray = np.empty((1000, 2))
    memory[:800] = rng.standard_normal((800, 2)) * 0.5
    memory[800:] = rng.standard_normal((200, 2)) * 3
    return normal, anomaly, memory


def _score_nn(test_pts: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Nearest-neighbor distance score for each test point."""
    return np.array([np.linalg.norm(bank - p, axis=1).min() for p in test_pts])


def regen_fig8_single_prototype() -> Path:
    """Generate Fig 8 v3 with fixed legend positioning.

    Left panel: Naive NN — test sample vs random memory bank. AUROC ~50%.
    Right panel: Single-prototype — test sample distance to centroid.
                 AUROC 99.70%.

    Returns the path to the generated PDF.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
    normal, anomaly, memory = _generate_data(seed=42)

    # ============ Left: Naive NN ============
    ax_left: plt.Axes = axes[0]
    nn_scores_normal = _score_nn(normal, memory)
    nn_scores_anomaly = _score_nn(anomaly, memory)
    auroc_left = _compute_auroc(nn_scores_normal, nn_scores_anomaly)

    ax_left.scatter(memory[:800, 0], memory[:800, 1], s=2, c="lightgray", alpha=0.4, label="Memory (1000)")
    ax_left.scatter(memory[800:, 0], memory[800:, 1], s=2, c="darkgray", alpha=0.5, label="Memory (outliers)")
    ax_left.scatter(normal[:, 0], normal[:, 1], s=25, c="steelblue", marker="o", edgecolors="black", linewidths=0.5, label="Normal test")
    ax_left.scatter(anomaly[:, 0], anomaly[:, 1], s=40, c="crimson", marker="X", edgecolors="black", linewidths=0.5, label="Anomaly test")
    ax_left.set_title(f"(a) Naive NN  —  AUROC={auroc_left:.2f}")
    ax_left.set_xlabel(r"$f_1$ (DINOv3 patch feature dim 1)")
    ax_left.set_ylabel(r"$f_2$ (DINOv3 patch feature dim 2)")
    ax_left.set_xlim(-5, 5)
    ax_left.set_ylim(-5, 7)
    # FIX: legend to lower right (data cluster is in upper-left, no overlap)
    ax_left.legend(loc="lower right", framealpha=0.9, edgecolor="gray", fontsize=8)
    ax_left.grid(True, alpha=0.2, linestyle=":")

    # ============ Right: Single-prototype ============
    ax_right: plt.Axes = axes[1]
    centroid: np.ndarray = memory.mean(axis=0)
    centroid_score_normal = np.linalg.norm(normal - centroid, axis=1)
    centroid_score_anomaly = np.linalg.norm(anomaly - centroid, axis=1)
    auroc_right = _compute_auroc(centroid_score_normal, centroid_score_anomaly)

    ax_right.scatter(memory[:800, 0], memory[:800, 1], s=2, c="lightgray", alpha=0.4, label="Memory (1000)")
    ax_right.scatter(memory[800:, 0], memory[800:, 1], s=2, c="darkgray", alpha=0.5, label="Memory (outliers)")
    ax_right.scatter(normal[:, 0], normal[:, 1], s=25, c="steelblue", marker="o", edgecolors="black", linewidths=0.5, label="Normal test")
    ax_right.scatter(anomaly[:, 0], anomaly[:, 1], s=40, c="crimson", marker="X", edgecolors="black", linewidths=0.5, label="Anomaly test")
    # centroid marker
    ax_right.scatter(*centroid, s=200, c="gold", marker="*", edgecolors="black", linewidths=0.8, label="Centroid c")
    ax_right.set_title(f"(b) Single prototype (centroid)  —  AUROC={auroc_right:.4f}")
    ax_right.set_xlabel(r"$f_1$ (DINOv3 patch feature dim 1)")
    ax_right.set_ylabel("")
    ax_right.set_xlim(-5, 5)
    ax_right.set_ylim(-5, 7)
    # FIX: legend in lower right (was upper right, occluded with annotation)
    ax_right.legend(loc="lower right", framealpha=0.9, edgecolor="gray", fontsize=8)
    ax_right.grid(True, alpha=0.2, linestyle=":")
    # FIX: green annotation moved to (x=-3.5, y=5.5) with white background bbox
    ax_right.annotate(
        "All anomalies are far\nfrom centroid (large score)",
        xy=(-3.5, 5.5),
        fontsize=8,
        color="darkgreen",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="green", boxstyle="round,pad=0.3"),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "figure_single_text_prototype.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    logger.info("Wrote %s (%.1f KB) — AUROC left=%.2f, right=%.4f", out_path, out_path.stat().st_size / 1024, auroc_left, auroc_right)
    plt.close(fig)
    return out_path


def _compute_auroc(scores_normal: np.ndarray, scores_anomaly: np.ndarray) -> float:
    """Compute AUROC using the Wilcoxon-Mann-Whitney statistic.

    AUROC = P(score(anomaly) > score(normal)).
    Higher scores = more anomalous.
    """
    n_n = len(scores_normal)
    n_a = len(scores_anomaly)
    # Vectorized: count pairs where anomaly_score > normal_score
    diff = scores_anomaly[:, None] - scores_normal[None, :]
    n_correct = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    return n_correct / (n_n * n_a)


if __name__ == "__main__":
    out = regen_fig8_single_prototype()
    logger.info("Done. fig8 v3 saved to %s", out)