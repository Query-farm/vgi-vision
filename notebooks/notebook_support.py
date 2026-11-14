"""Presentation helpers; score accounting and availability checks stay in visible SQL."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def theme():
    """Apply consistent readable chart defaults."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "text.parse_math": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def finish(fig, name, title, subtitle, provenance, left=0.1):
    """Display and save a figure with source attribution and units."""
    fig.suptitle(title, x=0.09, y=0.98, ha="left", fontsize=17, fontweight="bold", color="#243e32")
    fig.text(0.09, 0.90, subtitle, ha="left", fontsize=10.5, color="#526b5b")
    fig.text(
        0.09,
        0.025,
        provenance,
        fontsize=9,
        color="#526b5b",
    )
    fig.subplots_adjust(left=left, right=0.97, bottom=0.2, top=0.79, hspace=0.2)
    directory = Path("validation/charts")
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", bbox_inches="tight")
    plt.show()
