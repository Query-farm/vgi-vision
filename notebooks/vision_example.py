"""Verify image fixtures and render the actual worker's score distributions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from vgi_notebooks import finish

PROVENANCE = (
    "Inputs: Chelsea photo — Stefan van der Walt, CC0 / scikit-image; synthetic checkerboard — Query.Farm\n"
    "MobileNetV2-12 / ImageNet-1k · pinned ONNX weights · CPU inference via VGI Vision v0.4.0"
)


def validate_inputs(top_k):
    """Validate the bounded reporting parameter and immutable authored inputs."""
    if type(top_k) is not int or not 1 <= top_k <= 50:
        raise ValueError("TOP_K must be an integer from one through fifty")
    provenance = json.loads(Path("fixtures/provenance.json").read_text())
    if hashlib.sha256(Path("fixtures/chelsea.png").read_bytes()).hexdigest() != provenance["photo"]["sha256"]:
        raise ValueError("Photograph differs from its published bytes")
    for entry in provenance["synthetic"]:
        if hashlib.sha256((Path("fixtures") / entry["file"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Synthetic image fixture differs from its declared bytes")


def input_gallery():
    """Render measured evidence without recomputing classifier scores."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, path, title in zip(
        axes,
        ["chelsea.png", "checkerboard.png"],
        ["CC0 photograph · 451 × 300 pixels", "Synthetic checkerboard · 224 × 224 pixels"],
        strict=True,
    ):
        with Image.open(Path("fixtures") / path) as im:
            ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    finish(
        fig,
        "Two valid images, two different kinds of evidence",
        "Original inputs shown here · classifier resizes both to a 224 × 224 RGB tensor",
        PROVENANCE,
        name="01-input-images",
    )


def score_chart(heads):
    """Render measured evidence without recomputing classifier scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    for ax, image_id, title, color in zip(
        axes, ["photo", "pattern"], ["Cat photograph", "Synthetic checkerboard"], ["#397f91", "#ac6d48"], strict=True
    ):
        rows = heads.loc[heads.image_id == image_id].sort_values("score_rank", ascending=False)
        ax.barh(rows.label, 100 * rows.confidence, color=color, height=0.65)
        ax.set(xlim=(0, 100), xlabel="Softmax score · percent", title=title)
        ax.yaxis.grid(False)
        for i, score in enumerate(rows.confidence):
            ax.text(100 * score + 1, i, f"{100 * score:.1f}%", va="center", fontsize=9)
    finish(
        fig,
        "A valid image can receive a label without proving a match",
        "Top five original scores · same 0–100% scale · held fixed when TOP_K changes",
        PROVENANCE,
        name="02-top-five-reference",
    )


def mass_chart(mass, curves, TOP_K):
    """Render measured evidence without recomputing classifier scores."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for image_id, label, color, marker in [
        ("photo", "Cat photograph", "#397f91", "o"),
        ("pattern", "Synthetic checkerboard", "#ac6d48", "s"),
    ]:
        rows = curves.loc[curves.image_id == image_id]
        ax.plot(rows.score_rank, rows.cumulative_mass, color=color, label=label, linewidth=2)
        at = rows.loc[rows.score_rank == TOP_K].iloc[0]
        ax.scatter([TOP_K], [at.cumulative_mass], color=color, marker=marker, s=70, zorder=5)
    ax.axvline(TOP_K, color="#556b5b", linestyle="--", linewidth=1.2, label=f"Requested K = {TOP_K}")
    ax.set(
        xlim=(0, 50),
        ylim=(0, 1.03),
        xlabel="Number of highest-scoring class entries retained",
        ylabel="Cumulative softmax mass · fraction of 1",
    )
    ax.legend(loc="lower right", frameon=True)
    finish(
        fig,
        "A longer list reveals more of the same distribution",
        f"Markers show the requested {TOP_K} entries · remaining mass is omitted, not renormalized",
        PROVENANCE,
        name="03-retained-mass",
    )
