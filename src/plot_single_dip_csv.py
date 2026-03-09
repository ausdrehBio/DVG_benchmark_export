#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY_COLS = ["Segment", "Start", "End", "NGS_read_count"]
SEGMENT_ORDER = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]


def _pick_col(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Supported schemas:
    # 1) Segment, Start, End, NGS_read_count
    # 2) reference_segment|genome_id, predicted_start, predicted_end, [optional count]
    segment_col = _pick_col(df.columns.tolist(), ["Segment", "reference_segment", "genome_id"])
    start_col = _pick_col(df.columns.tolist(), ["Start", "predicted_start", "BP"])
    end_col = _pick_col(df.columns.tolist(), ["End", "predicted_end", "RI"])
    count_col = _pick_col(df.columns.tolist(), ["NGS_read_count", "read_count", "count"])

    missing = []
    if segment_col is None:
        missing.append("Segment/reference_segment/genome_id")
    if start_col is None:
        missing.append("Start/predicted_start/BP")
    if end_col is None:
        missing.append("End/predicted_end/RI")
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    work = df.copy()
    if count_col is None:
        work["__count__"] = 1
        count_col = "__count__"

    out = work[[segment_col, start_col, end_col, count_col]].copy()
    out.columns = KEY_COLS
    out["Segment"] = out["Segment"].astype(str).str.strip()
    out["Start"] = pd.to_numeric(out["Start"], errors="coerce")
    out["End"] = pd.to_numeric(out["End"], errors="coerce")
    out["NGS_read_count"] = pd.to_numeric(out["NGS_read_count"], errors="coerce")
    out = out.dropna(subset=["Start", "End", "NGS_read_count"])

    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    out["NGS_read_count"] = out["NGS_read_count"].astype(int)

    out = out.groupby(["Segment", "Start", "End"], as_index=False)["NGS_read_count"].sum()

    seg_norm = out["Segment"].astype(str).str.strip()
    invalid_segment = (seg_norm == "") | (seg_norm.str.lower().isin({"nan", "none", "<na>"}))
    out = out[~invalid_segment].copy()
    out = out[out["End"] > out["Start"]].copy()
    out["Deletion_length"] = out["End"] - out["Start"]
    return out


def infer_strain(path: Path) -> str | None:
    parts = list(path.resolve().parts)
    if "results" in parts:
        i = parts.index("results")
        if i + 2 < len(parts):
            return parts[i + 2]
    return None


def ordered_segments(series: pd.Series) -> list[str]:
    present = set(series.unique())
    ordered = [s for s in SEGMENT_ORDER if s in present]
    extras = sorted([s for s in present if s not in SEGMENT_ORDER])
    return ordered + extras


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def _weighted_density(values: np.ndarray, weights: np.ndarray, x_max: int, bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([0.0, float(x_max)]), np.array([0.0])
    edges = np.linspace(0, max(1, x_max), bins + 1)
    hist, edges = np.histogram(values, bins=edges, weights=weights, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, hist


def plot_arc_per_segment(seg_df: pd.DataFrame, sample: str, segment: str, out_png: Path, max_arcs: int) -> None:
    seq_max = int(seg_df["End"].max())
    counts = seg_df["NGS_read_count"].to_numpy(dtype=float)
    starts = seg_df["Start"].to_numpy(dtype=float)
    ends = seg_df["End"].to_numpy(dtype=float)

    x_s, d_s = _weighted_density(starts, counts, seq_max)
    x_e, d_e = _weighted_density(ends, counts, seq_max)
    d_max = max(float(d_s.max()) if len(d_s) else 0.0, float(d_e.max()) if len(d_e) else 0.0, 1e-6)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), tight_layout=True)
    fig.patch.set_facecolor("#05070d")
    ax.set_facecolor("#0b1133")

    bar_w = max(1.0, seq_max / 80.0)
    ax.bar(x_s, d_s, width=bar_w, color="#7aa2ff", alpha=0.65, edgecolor="none", label="Start density")
    ax.bar(x_e, -d_e, width=bar_w, color="#f4a6ff", alpha=0.65, edgecolor="none", label="End density")

    top = seg_df.sort_values("NGS_read_count", ascending=False).head(max_arcs).copy()
    cvals = np.log10(top["NGS_read_count"].to_numpy(dtype=float) + 1.0)
    cmin = float(cvals.min()) if len(cvals) else 0.0
    cmax = float(cvals.max()) if len(cvals) else 1.0
    crange = max(cmax - cmin, 1e-9)

    for row, c in zip(top.itertuples(index=False), cvals):
        s = float(row.Start)
        e = float(row.End)
        t = np.linspace(0.0, 1.0, 80)
        x = s + (e - s) * t

        span_frac = (e - s) / max(1.0, float(seq_max))
        base_h = d_max * (0.8 + 5.0 * span_frac)
        y = 4.0 * base_h * t * (1.0 - t)

        strength = (c - cmin) / crange
        lw = 0.35 + 1.7 * strength
        alpha = 0.12 + 0.35 * strength
        ax.plot(x, y, color="#39d62f", linewidth=lw, alpha=alpha)

    y_low = -d_max * 1.25
    y_high = d_max * 7.2
    ax.set_ylim(y_low, y_high)
    ax.set_xlim(0, seq_max)

    ax.axhline(0, color="white", linewidth=0.7, alpha=0.6)
    ax.grid(True, axis="x", color="white", alpha=0.12, linewidth=0.5)

    ax.set_xlabel("Nucleotide position", color="white")
    ax.set_ylabel("Probability density", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#9ab0ff")

    n_total = len(seg_df)
    n_plot = len(top)
    ax.set_title(
        f"{sample} | {segment}  (n={n_total}; arcs shown={n_plot})",
        color="white",
        fontsize=12,
    )

    leg = ax.legend(loc="upper right", framealpha=0.15, facecolor="#0b1133", edgecolor="#9ab0ff")
    for txt in leg.get_texts():
        txt.set_color("white")

    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_vertical_length_per_segment(seg_df: pd.DataFrame, sample: str, segment: str, out_png: Path) -> None:
    lengths = seg_df["Deletion_length"].to_numpy(dtype=float)
    n = len(lengths)

    fig, ax = plt.subplots(figsize=(3.3, 6.2), tight_layout=True)
    fig.patch.set_facecolor("#05070d")
    ax.set_facecolor("#0b1133")

    vp = ax.violinplot([lengths], positions=[1], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("#7aa2ff")
        body.set_edgecolor("#d6e2ff")
        body.set_alpha(0.35)

    bp = ax.boxplot([lengths], positions=[1], widths=0.23, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="#9cc1ff", alpha=0.55, edgecolor="#f2f5ff")
    for key in ["whiskers", "caps", "medians"]:
        for item in bp[key]:
            item.set(color="#f2f5ff", linewidth=1.1)

    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=1.0, scale=0.045, size=n)
    ax.scatter(jitter, lengths, s=10, alpha=0.45, color="#dbe6ff", edgecolors="none")

    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1])
    ax.set_xticklabels([f"{sample}\\n{segment} (n={n})"], color="white", rotation=90)
    ax.set_ylabel("DelVG sequence length (nt)", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, axis="y", color="white", alpha=0.12)

    for spine in ax.spines.values():
        spine.set_color("#9ab0ff")

    plt.savefig(out_png, dpi=180)
    plt.close()


def write_segment_summary(df: pd.DataFrame, out_csv: Path) -> None:
    rows = []
    for seg in ordered_segments(df["Segment"]):
        seg_df = df[df["Segment"] == seg]
        rows.append(
            {
                "Segment": seg,
                "n_junctions": int(len(seg_df)),
                "sum_ngs_read_count": int(seg_df["NGS_read_count"].sum()),
                "median_ngs_read_count": float(seg_df["NGS_read_count"].median()),
                "mean_delvg_length": float(seg_df["Deletion_length"].mean()),
                "median_delvg_length": float(seg_df["Deletion_length"].median()),
                "max_end_position": int(seg_df["End"].max()),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create DelVG arc + vertical-length plots per segment for one CSV."
    )
    parser.add_argument("csv", help="Input CSV path")
    parser.add_argument("--strain", default="", help="Strain name for output folder (default: inferred from path)")
    parser.add_argument("--sample", default="", help="Sample name for output folder and titles (default: CSV stem)")
    parser.add_argument(
        "--out-root",
        default="",
        help="Root output folder. Default: <DI_Pipeline_2>/results/plot",
    )
    parser.add_argument(
        "--max-arcs",
        type=int,
        default=2500,
        help="Maximum number of arcs to draw per segment (highest NGS_read_count first).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")

    strain = args.strain.strip() or infer_strain(csv_path)
    if not strain:
        raise SystemExit("Unable to infer strain from path. Pass --strain explicitly.")

    sample = args.sample.strip() or csv_path.stem

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = Path(__file__).resolve().parents[1] / "results" / "plot"

    out_dir = out_root / strain / safe_name(sample)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8")
    df = load_csv(csv_path)
    if df.empty:
        raise SystemExit("No valid rows after filtering (need End > Start and numeric required columns).")

    segments = ordered_segments(df["Segment"])
    for seg in segments:
        seg_df = df[df["Segment"] == seg].copy()
        seg_tag = safe_name(seg)
        plot_arc_per_segment(seg_df, sample, seg, out_dir / f"{seg_tag}_arc_plot.png", max_arcs=args.max_arcs)
        plot_vertical_length_per_segment(seg_df, sample, seg, out_dir / f"{seg_tag}_length_vertical.png")

    # Also generate the same two plots on the full dataset (ignoring segment).
    plot_arc_per_segment(df, sample, "ALL", out_dir / "ALL_arc_plot.png", max_arcs=args.max_arcs)
    plot_vertical_length_per_segment(df, sample, "ALL", out_dir / "ALL_length_vertical.png")

    write_segment_summary(df, out_dir / "segment_summary.csv")

    print(f"Plots saved to: {out_dir}")
    print(f"Segments plotted: {', '.join(segments)}")


if __name__ == "__main__":
    main()
