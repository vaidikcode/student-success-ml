"""
Exploratory Data Analysis with Matplotlib + Seaborn. Every figure is saved to figures/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
PALETTE = {"Passed": "#2a9d8f", "Failed": "#e76f51"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] saved figures/{name}")


def plot_distributions(df) -> None:
    """Matplotlib subplots grid: histograms of key features."""
    cols = ["study_hours_week", "sleep_hours", "attendance", "screen_time_hours", "prior_gpa", "final_score"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col, color in zip(axes.ravel(), cols, sns.color_palette("viridis", len(cols))):
        ax.hist(df[col], bins=30, color=color, edgecolor="white", alpha=0.9)
        ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1, label=f"mean = {df[col].mean():.1f}")
        ax.set_title(col.replace("_", " ").title())
        ax.legend(fontsize=8)
    fig.suptitle("Feature distributions", fontsize=15, weight="bold")
    _save(fig, "01_distributions.png")


def plot_correlation_heatmap(df, features) -> None:
    corr = df[features + ["final_score"]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)       # hide the duplicate upper triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.7})
    ax.grid(False)
    ax.set_title("Correlation matrix (what moves together?)", fontsize=14, weight="bold")
    _save(fig, "02_correlation_heatmap.png")


def plot_score_by_category(df) -> None:
    """Boxplot + violin plot: spot outliers and compare groups."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    order = df.groupby("track")["final_score"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="track", y="final_score", order=order, hue="track", palette="Set2", legend=False, ax=axes[0])
    axes[0].axhline(60, color="red", linestyle=":", label="pass line (60)")
    axes[0].set_title("Final score by track (boxplot)")
    axes[0].legend()

    plot_df = df.assign(job=df["part_time_job"].map({0: "No job", 1: "Part-time job"}))
    sns.violinplot(data=plot_df, x="year_label", y="final_score", hue="job", split=True, inner="quart",
                   order=["Freshman", "Sophomore", "Junior", "Senior"], palette="pastel", ax=axes[1])
    axes[1].set_title("Final score by year & job (violin plot)")
    axes[1].set_xlabel("")
    _save(fig, "03_score_by_category.png")


def plot_pass_fail_hist(df) -> None:
    plot_df = df.assign(result=df["passed"].map({1: "Passed", 0: "Failed"}))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in zip(axes, ["study_hours_week", "sleep_hours", "attendance"]):
        sns.histplot(data=plot_df, x=col, hue="result", kde=True, palette=PALETTE, element="step", ax=ax)
        ax.set_title(f"{col.replace('_', ' ')} — passed vs failed")
    _save(fig, "04_pass_fail_histograms.png")


def plot_pairplot(df) -> None:
    plot_df = df.assign(result=df["passed"].map({1: "Passed", 0: "Failed"}))
    g = sns.pairplot(plot_df[["study_hours_week", "sleep_hours", "attendance", "prior_gpa", "result"]].sample(500, random_state=0),
                     hue="result", palette=PALETTE, plot_kws={"alpha": 0.6, "s": 18}, corner=True)
    g.figure.suptitle("Pairplot of top features", y=1.02, fontsize=14, weight="bold")
    _save(g.figure, "05_pairplot.png")


def plot_relationships(df) -> None:
    """Scatter + bar + line in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    sc = axes[0].scatter(df["study_hours_week"], df["final_score"], c=df["sleep_hours"], cmap="coolwarm", s=12, alpha=0.7)
    fig.colorbar(sc, ax=axes[0], label="sleep hours")
    axes[0].set(xlabel="study hours / week", ylabel="final score", title="Study vs score (color = sleep)")

    sleep_bins = df.groupby(df["sleep_hours"].round())["final_score"].mean()
    axes[1].bar(sleep_bins.index, sleep_bins.values, color="#264653")
    axes[1].set(xlabel="sleep hours (rounded)", ylabel="avg final score", title="The sleep sweet spot", ylim=(40, 75))

    coffee = df.groupby("coffee_cups_day")["final_score"].agg(["mean", "count"]).query("count >= 10")
    axes[2].plot(coffee.index, coffee["mean"], marker="o", color="#6f4e37", linewidth=2.5)
    axes[2].set(xlabel="coffee cups / day", ylabel="avg final score", title="Coffee: helpful... up to a point")
    _save(fig, "06_relationships.png")


def run_eda(df, features) -> None:
    print("\n=== EDA: generating figures ===")
    plot_distributions(df)
    plot_correlation_heatmap(df, [f for f in features if not f.startswith("track_")])
    plot_score_by_category(df)
    plot_pass_fail_hist(df)
    plot_pairplot(df)
    plot_relationships(df)
