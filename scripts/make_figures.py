"""Regenerate every figure in README from raw results. Usage: python scripts/make_figures.py"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catadye.metrics import forgetting, new_domain_loss  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
COL = {"none": "#6b7280", "cms": "#d9480f", "lr_matched": "#1c7ed6"}
LAB = {"none": "baseline", "cms": "CMS 1/8/64", "lr_matched": "LR-matched"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def load(d):
    runs = []
    for f in sorted((ROOT / "results" / d).glob("*.json")):
        runs.append(json.loads(f.read_text()))
    return runs


# ---- 1. update schedule diagram ------------------------------------------------------------
def fig_schedule():
    T = 96
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.6), sharex=True, gridspec_kw={"hspace": 0.5})
    for ax, mode in zip(axes, ["CMS (staggered)", "LR-matched control"]):
        for row, (p, name) in enumerate([(1, "sub-block 1 (period 1)"), (8, "sub-block 2 (period 8)"),
                                         (64, "sub-block 3 (period 64)")]):
            y = 2 - row
            if mode.startswith("CMS"):
                ups = list(range(p, T + 1, p))
            else:
                ups = list(range(1, T + 1))
            ax.vlines(ups, y - 0.3, y + 0.3, color=COL["cms" if mode.startswith("CMS") else "lr_matched"],
                      lw=1.2 if p > 1 or not mode.startswith("CMS") else 0.8)
            if mode.startswith("LR") and p > 1:
                ax.text(T + 1, y, f"lr / {p}", va="center", fontsize=8, color="#444")
            ax.text(-1.5, y, name, ha="right", va="center", fontsize=8)
        ax.set_yticks([])
        ax.set_title(mode + (": gradients accumulate, then one update" if mode.startswith("CMS")
                             else ": every step, but smaller step size"), fontsize=9, loc="left")
        ax.set_xlim(0, T + 8)
    axes[1].set_xlabel("optimizer step (tick = parameters updated)")
    fig.savefig(FIG / "update_schedule.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---- 2. pilot 1 results --------------------------------------------------------------------
def fig_pilot1(runs):
    by = defaultdict(list)
    for r in runs:
        R = np.array(r["R"])
        by[r["mode"]].append((forgetting(R), new_domain_loss(R), r["R"]))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"wspace": 0.38})
    arms = ["none", "cms", "lr_matched"]
    for ax, idx, title in ((axes[0], 0, "Forgetting (nats, lower = better)"),
                           (axes[1], 1, "New-domain loss (plasticity)")):
        for i, a in enumerate(arms):
            v = np.array([t[idx] for t in by[a]])
            ax.bar(i, v.mean(), color=COL[a], alpha=0.35, width=0.6)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[a], s=22, zorder=3)
        ax.set_xticks(range(3), [LAB[a] for a in arms], fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylim(0, None)
    axes[1].set_ylim(2.4, 2.9)
    ax = axes[2]
    for a in arms:
        pts = np.array([[t[1], t[0]] for t in by[a]])
        ax.scatter(pts[:, 0], pts[:, 1], color=COL[a], label=LAB[a], s=28)
    ax.set_xlabel("new-domain loss")
    ax.set_ylabel("forgetting")
    ax.set_title("Per-seed points", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Pilot 1 (16.1M params, TinyStories -> Python, 3 seeds): CMS did not reduce forgetting",
                 fontsize=10, y=1.03)
    fig.savefig(FIG / "pilot1_results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---- 3. revised pilot: stability-plasticity frontier + order check ------------------------------
def fig_rev(runs):
    A = [r for r in runs if r["cfg"].get("order") == "A"]
    B = [r for r in runs if r["cfg"].get("order") == "B"]
    if not A:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for a in ["none", "cms", "lr_matched"]:
        lrs = sorted({r["cfg"]["lr"] for r in A if r["mode"] == a})
        xs, ys, xe, ye = [], [], [], []
        for lr in lrs:
            v = [(new_domain_loss(np.array(r["R"])), forgetting(np.array(r["R"])))
                 for r in A if r["mode"] == a and r["cfg"]["lr"] == lr]
            v = np.array(v)
            xs.append(v[:, 0].mean()); ys.append(v[:, 1].mean())
            xe.append(v[:, 0].std(ddof=1) if len(v) > 1 else 0); ye.append(v[:, 1].std(ddof=1) if len(v) > 1 else 0)
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=COL[a], marker="o", ms=5, lw=1.4, capsize=2, label=LAB[a])
        for x, y, lr in zip(xs, ys, lrs):
            ax.annotate(f"{lr:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6.5, color=COL[a])
    ax.set_xlabel("new-domain loss (plasticity; lower = learns more)")
    ax.set_ylabel("forgetting (nats; lower = better)")
    ax.set_title("Stability-plasticity frontier, order A (labels = lr)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    if B:
        for i, a in enumerate(["none", "cms", "lr_matched"]):
            v = np.array([forgetting(np.array(r["R"])) for r in B if r["mode"] == a])
            if len(v):
                ax.bar(i, v.mean(), color=COL[a], alpha=0.35, width=0.6)
                ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[a], s=22, zorder=3)
        ax.set_xticks(range(3), [LAB[a] for a in ["none", "cms", "lr_matched"]], fontsize=8)
    ax.set_title("Reversed order B (Python -> TinyStories), lr 6e-4", fontsize=9)
    ax.set_ylabel("forgetting")
    share = np.mean([r["sub_param_share"] for r in runs])
    fig.suptitle(f"Revised pilot ({runs[0]['n_params']/1e6:.1f}M params, staggered share {share:.0%})",
                 fontsize=10, y=1.03)
    fig.savefig(FIG / "pilot_rev_frontier.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


# ---- 4. how forgetting is measured (real numbers) ---------------------------------------------
def fig_loss_matrix(runs):
    A = [r for r in runs if r["cfg"].get("order") == "A" and r["mode"] == "none" and r["cfg"]["lr"] == 6e-4]
    if not A:
        return False
    R = np.mean([r["R"] for r in A], axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.grid(False)
    im = ax.imshow(R, cmap="Oranges", vmin=2, vmax=7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["TinyStories\n(old)", "Python\n(new)"])
    ax.set_yticks([0, 1], ["after training\non TinyStories", "after training\non Python"])
    ax.set_xlabel("evaluated on", fontsize=9)
    ax.annotate("", xy=(0.40, 0.78), xytext=(0.40, 0.22), arrowprops=dict(arrowstyle="->", lw=2, color="#c92a2a"))
    ax.annotate(f"forgetting\n= R[1,0] - R[0,0]\n= +{R[1, 0] - R[0, 0]:.2f} nats", xy=(1.04, 0.5),
                xycoords="axes fraction", color="#c92a2a", fontsize=9, va="center", ha="left")
    ax.set_title("Loss matrix R[i,j] (baseline, lr 6e-4, mean of 3 seeds, nats)", fontsize=9)
    fig.savefig(FIG / "loss_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


# ---- 5. matched plasticity on the real frontier ------------------------------------------------
def fig_matched(runs):
    A = [r for r in runs if r["cfg"].get("order") == "A"]
    if not A:
        return False
    pts = {}
    for a in ["none", "cms", "lr_matched"]:
        v = []
        for lr in sorted({r["cfg"]["lr"] for r in A}):
            rr = [(new_domain_loss(np.array(r["R"])), forgetting(np.array(r["R"])))
                  for r in A if r["mode"] == a and r["cfg"]["lr"] == lr]
            v.append(np.mean(rr, axis=0))
        pts[a] = np.array(sorted(v, key=lambda t: t[0]))
    target = float(np.median([new_domain_loss(np.array(r["R"])) for r in A if r["mode"] == "none"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    vals = {}
    for a, p in pts.items():
        ax.plot(p[:, 0], p[:, 1], "-o", color=COL[a], ms=4, lw=1.2, alpha=0.8, label=LAB[a])
        f = float(np.interp(target, p[:, 0], p[:, 1]))
        ax.scatter([target], [f], s=90, facecolor="white", edgecolor=COL[a], lw=2, zorder=5)
        vals[a] = f
    ax.axvline(target, color="#444", ls="--", lw=1)
    ax.text(target + 0.008, 4.27, f"target plasticity\n(median baseline = {target:.2f})", fontsize=8, va="top",
            color="#444")
    ax.text(0.985, 0.97, "forgetting at the target (circles):\n" + "\n".join(
        f"{LAB[a]:11s} {vals[a]:.2f}" for a in ("none", "cms", "lr_matched")) +
        "\n\nillustrative: interpolated on seed-averaged\ncurves; the pre-registered per-seed test\ngives 3.59 / 3.68 / 3.57",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.set_xlim(2.55, 3.05); ax.set_ylim(3.1, 4.3)
    ax.set_xlabel("new-domain loss (plasticity)")
    ax.set_ylabel("forgetting (nats)")
    ax.set_title("Matched plasticity: read each arm's forgetting where it crosses the dashed line", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(FIG / "matched_plasticity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


# ---- 6. post-hoc drift diagnostic ----------------------------------------------------------------
def fig_drift():
    f = ROOT / "results" / "diagnostics.json"
    if not f.exists():
        return False
    D = json.loads(f.read_text())
    groups = ["embeddings", "attention", "layernorm", "mlp_sub0", "mlp_sub1", "mlp_sub2"]
    names = ["embeddings", "attention", "layernorm", "MLP sub-block 1\n(period 1)", "MLP sub-block 2\n(period 8)",
             "MLP sub-block 3\n(period 64)"]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    w = 0.27
    for k, (mode, lab) in enumerate([("none", "baseline"), ("cms", "CMS 1/8/64"), ("lr_matched", "LR-matched")]):
        rows = [d for d in D if d["mode"] == mode]
        v = np.array([[d["drift"][g] for g in groups] for d in rows])
        fg = np.mean([d["forgetting"] for d in rows])
        ax.bar(np.arange(6) + (k - 1) * w, v.mean(0), w, color=COL[mode], alpha=0.85,
               label=f"{lab} (forgetting {fg:.2f}, n={len(rows)})")
    ax.set_xticks(range(6), names, fontsize=8)
    ax.set_ylabel("relative weight drift\n||theta_after - theta_before|| / ||theta_before||")
    ax.set_title("Post-hoc: slow MLP blocks barely move under CMS / LR-matched, yet forgetting is unchanged",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "drift_diagnostic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


# ---- 7. Stage 1b test-time probe -----------------------------------------------------------------
def fig_ttt():
    runs = load("ttt")
    if len(runs) < 30:
        return False
    arms = {"ttt_single": ("#6b7280", "single-rate updates"), "ttt_cms": (COL["cms"], "CMS 1/8/64"),
            "ttt_lrm": (COL["lr_matched"], "LR-matched")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for a, (c, lab) in arms.items():
        pts = []
        for lr in (1e-4, 3e-4, 1e-3):
            v = [(r["curve"][-1]["new"], r["curve"][-1]["old"] - r["curve"][0]["old"])
                 for r in runs if r["arm"] == a and r["lr"] == lr]
            pts.append((np.mean([t[0] for t in v]), np.mean([t[1] for t in v]), lr))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c, ms=5, label=lab)
        for x, y, lr in pts:
            ax.annotate(f"{lr:g}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=c)
    ax.set_xlabel("final Python loss (plasticity)")
    ax.set_ylabel("TinyStories loss increase (forgetting, nats)")
    ax.set_title("Same learning rate: CMS forgets less than single-rate,\nLR-matched forgets less still (labels = lr)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for a, (c, lab) in arms.items():
        cs = [r["curve"] for r in runs if r["arm"] == a and r["lr"] == 3e-4]
        steps = [p["step"] for p in cs[0]]
        ax.plot(steps, [np.mean([cv[i]["old"] for cv in cs]) for i in range(len(steps))], "-o", color=c, ms=4,
                label=lab)
    ax.set_xlabel("test-time steps (4 x 256 tokens each)")
    ax.set_ylabel("held-out TinyStories loss")
    ax.set_title("Old-domain loss while adapting to Python (lr 3e-4, 3 seeds)", fontsize=9)
    fig.suptitle("Stage 1b: test-time updates of the MLP blocks on a Python stream", fontsize=10, y=1.03)
    fig.savefig(FIG / "ttt_probe.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    print("ttt fig:", fig_ttt())
    print("drift fig:", fig_drift())
    _rev = load("pilot_rev")
    print("loss matrix:", fig_loss_matrix(_rev) if _rev else False, "; matched:", fig_matched(_rev) if _rev else False)
    fig_schedule()
    p1 = load("pilot")
    if p1:
        fig_pilot1(p1)
    rev = load("pilot_rev")
    print("schedule: ok; pilot1:", bool(p1), "; rev runs:", len(rev), "; rev fig:", fig_rev(rev) if rev else False)
