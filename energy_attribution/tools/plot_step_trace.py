"""Plot a step trace (and optionally its manifest) for visual sanity checks.

    python energy_attribution/tools/plot_step_trace.py \
        --trace energy_attribution/runs/<run>/step_trace.*.jsonl \
        --manifest energy_attribution/runs/<run>/manifest.json

Writes PNGs next to the trace (or into --out):

1. request_timeline.png: per request, when it was prefilling vs decoding.
2. step_timeline.png: batch composition, new tokens and duration per step.
3. step_duration.png: step duration against the work in the step.
4. trace_vs_manifest.png: per-request token counts, trace vs server report
   (needs --manifest).

Needs pandas and matplotlib.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Categorical slots 1-3 of the reference palette (validated all-pairs for CVD).
PREFILL = "#2a78d6"
DECODE = "#eb6834"
MIXED = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
KIND_COLORS = {"prefill": PREFILL, "decode": DECODE, "mixed": MIXED}
KIND_LABELS = {"prefill": "Prefill only", "decode": "Decode only", "mixed": "Mixed"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    }
)


IDLE_GAP_S = 0.5


def load_trace(path, rids=None):
    records = {"meta": [], "anchor": [], "step": [], "owner": []}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["kind"]].append(rec)
    steps = pd.DataFrame(records["step"]).sort_values("step_id")
    owners = pd.DataFrame(records["owner"])
    gap_s = (steps["t_start_ns"] - steps["t_end_ns"].shift()) / 1e9
    steps["after_idle"] = gap_s.fillna(np.inf) >= IDLE_GAP_S
    if rids is not None:
        # Keep the workload window: from the first to the last step of a manifest request.
        workload = owners.loc[owners["rid"].isin(rids), "step_id"]
        steps = steps[steps.step_id.between(workload.min(), workload.max())]
        owners = owners[owners["rid"].isin(rids)]
    t0 = steps["t_start_ns"].min()
    steps["t0_s"] = (steps["t_start_ns"] - t0) / 1e9
    steps["t1_s"] = (steps["t_end_ns"] - t0) / 1e9
    steps["dur_ms"] = (steps["t_end_ns"] - steps["t_start_ns"]) / 1e6
    steps["kind"] = np.select(
        [
            (steps.num_prefill_reqs > 0) & (steps.num_decode_reqs > 0),
            steps.num_prefill_reqs > 0,
        ],
        ["mixed", "prefill"],
        default="decode",
    )
    owners = owners.merge(steps[["step_id", "t0_s", "t1_s"]], on="step_id")
    return records["meta"][0], steps, owners


def kind_legend(ax, kinds, loc="upper right"):
    handles = [
        Line2D([], [], color=KIND_COLORS[k], lw=6, label=KIND_LABELS[k]) for k in kinds
    ]
    ax.legend(handles=handles, loc=loc, ncols=len(handles))


def plot_request_timeline(steps, owners, out):
    first = owners.groupby("rid")["t0_s"].min().sort_values()
    row = {rid: i for i, rid in enumerate(first.index)}
    n = len(row)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.045 * n + 1.5)))
    lw = max(0.6, min(4.0, 260 / n))
    dec = owners[owners.phase == "decode"]
    y = dec["rid"].map(row).to_numpy()
    segs = np.stack([np.c_[dec["t0_s"], y], np.c_[dec["t1_s"], y]], axis=1)
    ax.add_collection(LineCollection(segs, colors=DECODE, linewidths=lw))
    # Prefill steps last ~20 ms, invisible as segments at this scale; draw ticks.
    pre = owners[owners.phase == "prefill"]
    ax.scatter(
        (pre["t0_s"] + pre["t1_s"]) / 2, pre["rid"].map(row), marker="|",
        s=(lw * 4) ** 2, color=PREFILL, lw=1.6, zorder=3,
    )
    ax.set_xlim(0, steps["t1_s"].max() * 1.01)
    ax.set_ylim(n, -1)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time since first step (s)")
    ax.set_ylabel(f"Requests ({n}), by first step")
    ax.set_title("Request timeline: GPU steps each request took part in")
    ax.legend(
        handles=[
            Line2D([], [], color=PREFILL, lw=6, label="Prefill"),
            Line2D([], [], color=DECODE, lw=6, label="Decode"),
        ],
        loc="upper right",
        ncols=2,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "request_timeline.png"), dpi=150)
    plt.close(fig)


def plot_step_timeline(steps, out):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    x, w = steps["t0_s"], steps["t1_s"] - steps["t0_s"]

    ax = axes[0]
    ax.bar(x, steps.num_prefill_reqs, width=w, align="edge", color=PREFILL, lw=0)
    ax.bar(
        x, steps.num_decode_reqs, width=w, bottom=steps.num_prefill_reqs,
        align="edge", color=DECODE, lw=0,
    )
    ax.set_ylabel("Requests in step")
    ax.set_title("Batch composition per step")
    kind_legend(ax, ["prefill", "decode"])

    ax = axes[1]
    ax.bar(x, steps.prefill_tokens, width=w, align="edge", color=PREFILL, lw=0)
    ax.bar(
        x, steps.decode_tokens, width=w, bottom=steps.prefill_tokens,
        align="edge", color=DECODE, lw=0,
    )
    ax.set_ylabel("New tokens in step")
    ax.set_title("New tokens computed per step")

    ax = axes[2]
    for kind in ("decode", "prefill", "mixed"):
        sub = steps[steps.kind == kind]
        ax.scatter(sub.t0_s, sub.dur_ms, s=6, color=KIND_COLORS[kind], lw=0)
    idle = steps[steps.after_idle]
    ax.scatter(idle.t0_s, idle.dur_ms, s=60, facecolors="none", edgecolors=INK, lw=1)
    ax.set_ylabel("GPU window (ms)")
    ax.set_xlabel("Time since first step (s)")
    ax.set_title(f"Step duration (ringed: first step after {IDLE_GAP_S:g} s or more idle)")
    kind_legend(ax, [k for k in ("prefill", "decode", "mixed") if (steps.kind == k).any()])

    fig.tight_layout()
    fig.savefig(os.path.join(out, "step_timeline.png"), dpi=150)
    plt.close(fig)


def plot_step_duration(steps, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    dec = steps[steps.kind == "decode"]
    sc = ax.scatter(
        dec.batch_size, dec.dur_ms, c=dec.kv_read_tokens, s=8, lw=0, cmap="Blues",
        vmin=0,
    )
    fig.colorbar(sc, ax=ax, label="KV tokens read")
    idle = dec[dec.after_idle]
    ax.scatter(idle.batch_size, idle.dur_ms, s=60, facecolors="none", edgecolors=INK, lw=1)
    ax.set_xlabel("Requests in step")
    ax.set_ylabel("GPU window (ms)")
    ax.set_title("Decode-only steps")

    ax = axes[1]
    for kind in ("prefill", "mixed"):
        sub = steps[steps.kind == kind]
        ax.scatter(sub.prefill_tokens, sub.dur_ms, s=10, color=KIND_COLORS[kind], lw=0)
    idle = steps[steps.kind.isin(["prefill", "mixed"]) & steps.after_idle]
    ax.scatter(idle.prefill_tokens, idle.dur_ms, s=60, facecolors="none", edgecolors=INK, lw=1)
    ax.set_xlabel("Prefill tokens in step")
    ax.set_ylabel("GPU window (ms)")
    ax.set_title("Prefill and mixed steps")
    kind_legend(ax, [k for k in ("prefill", "mixed") if (steps.kind == k).any()], "upper left")

    fig.text(
        0.01, 0.01, f"Ringed: first step after {IDLE_GAP_S:g} s or more of GPU idle.",
        color=INK_2, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "step_duration.png"), dpi=150)
    plt.close(fig)


def per_request(owners):
    rows = []
    for rid, g in owners.sort_values("step_id").groupby("rid"):
        pre = g[g.phase == "prefill"]
        rows.append(
            {
                "rid": rid,
                "computed": int(pre.selected_tokens.sum()),
                "reused": int(pre.cached_tokens.iloc[0]) if len(pre) else 0,
                "decode_rows": int((g.phase == "decode").sum()),
                "retracted": bool(g.retraction_count.max() > 0),
            }
        )
    return pd.DataFrame(rows)


def plot_trace_vs_manifest(owners, manifest_path, out, overlap=False):
    with open(manifest_path) as f:
        man = pd.DataFrame(json.load(f))
    df = man.merge(per_request(owners), on="rid", how="left")
    missing = int(df.computed.isna().sum())
    df = df.dropna(subset=["computed"])
    panels = [
        ("prompt_tokens", df.computed + df.reused, "Prompt tokens",
         "Trace: computed + reused"),
        ("cached_tokens", df.reused, "Cached prompt tokens", "Trace: reused"),
        # Under overlap a finished request rides along for one discarded step.
        ("completion_tokens", df.decode_rows + 1 - int(overlap), "Completion tokens",
         "Trace: decode steps" if overlap else "Trace: decode steps + 1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (col, ours, title, ylabel) in zip(axes, panels):
        ref = df[col]
        hi = max(ref.max(), ours.max()) * 1.05 or 1
        ax.plot([0, hi], [0, hi], color=INK_2, lw=1, ls="--", zorder=0)
        ok = ~df.retracted
        ax.scatter(ref[ok], ours[ok], s=14, color=PREFILL, lw=0)
        if (~ok).any():
            ax.scatter(ref[~ok], ours[~ok], s=18, color=DECODE, lw=0)
        bad = int(((ref != ours) & ok).sum())
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Server report: {col}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(
            0.04, 0.96, f"{len(df) - bad}/{len(df)} on the diagonal",
            transform=ax.transAxes, va="top", color=INK, fontsize=9,
        )
    note = "Dashed line: trace equals server report."
    if (df.retracted).any():
        note += " Orange: retracted requests (expected off-diagonal)."
    if missing:
        note += f" {missing} manifest requests missing from the trace."
    fig.text(0.01, 0.01, note, color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "trace_vs_manifest.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="trace JSONL (a glob is fine)")
    p.add_argument("--manifest", help="manifest.json from step_trace_client.py send")
    p.add_argument("--out", help="output directory (default: next to the trace)")
    args = p.parse_args()

    paths = sorted(glob.glob(args.trace))
    if len(paths) != 1:
        raise SystemExit(f"--trace must match exactly one file, got {paths}")
    trace = paths[0]
    out = args.out or os.path.dirname(os.path.abspath(trace))
    os.makedirs(out, exist_ok=True)

    rids = None
    if args.manifest:
        with open(args.manifest) as f:
            rids = {r["rid"] for r in json.load(f)}
    meta, steps, owners = load_trace(trace, rids=rids)
    plot_request_timeline(steps, owners, out)
    plot_step_timeline(steps, out)
    plot_step_duration(steps, out)
    if args.manifest:
        plot_trace_vs_manifest(owners, args.manifest, out, overlap=meta["enable_overlap"])
    print(
        f"{meta['device_name']}: {len(steps)} steps, {len(owners)} owner rows; "
        f"figures in {out}"
    )


if __name__ == "__main__":
    main()
