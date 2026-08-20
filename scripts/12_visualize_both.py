#!/usr/bin/env python
"""
스크립트 12: 두 데이터셋의 unfaithfulness rate를 한 그림에 (Figure 2).

VinDr-CXR와 NIH ChestX-ray14를 좌우 패널로 배치하여
데이터셋 간 일반화를 시각적으로 보인다.

사용:
    python scripts/12_visualize_both.py
"""
import json
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_stats(path):
    """(model, test, condition) -> (n_flip, n_unfaith, rate, lo, hi)"""
    grp = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["test_name"] == "masking":
                continue
            k = (r["model_id"], r["test_name"], r["prompt_condition"])
            if r["prediction_flipped"]:
                grp[k].append(1 if r["is_unfaithful"] else 0)

    out = {}
    for k, vals in grp.items():
        n = len(vals)
        if n == 0:
            continue
        rate = sum(vals) / n
        rng = random.Random(0)
        boots = []
        for _ in range(2000):
            s = [vals[rng.randrange(n)] for _ in range(n)]
            boots.append(sum(s) / n)
        boots.sort()
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots))]
        out[k] = (n, sum(vals), rate, lo, hi)
    return out


def draw_panel(ax, stats, title, show_ylabel):
    order = [
        ("counterfactual", "cot", "CF\nCoT"),
        ("counterfactual", "posthoc", "CF\nPost"),
        ("biasing", "cot", "Bias\nCoT"),
        ("biasing", "posthoc", "Bias\nPost"),
    ]
    models = [("gpt54", "GPT-5.4", "#378ADD", "#185FA5"),
              ("medgemma4b", "MedGemma-4B", "#1D9E75", "#0F6E56")]

    x = np.arange(len(order))
    width = 0.36

    for i, (mid, mlabel, fill, edge) in enumerate(models):
        vals, los, his = [], [], []
        for test, cond, _ in order:
            s = stats.get((mid, test, cond))
            if s is None:
                vals.append(0); los.append(0); his.append(0)
            else:
                _, _, rate, lo, hi = s
                vals.append(rate)
                los.append(max(0, rate - lo))
                his.append(max(0, hi - rate))
        pos = x + (i - 0.5) * width
        ax.bar(pos, vals, width, label=mlabel if show_ylabel else None,
               color=fill, edgecolor=edge, linewidth=0.6)
        ax.errorbar(pos, vals, yerr=[los, his], fmt="none",
                    ecolor=edge, elinewidth=0.9, capsize=3)

        # n 표시
        for j, (test, cond, _) in enumerate(order):
            s = stats.get((mid, test, cond))
            if s:
                ax.annotate(f"n={s[0]}", xy=(pos[j], 0.02),
                            ha="center", fontsize=7, color="white"
                            if vals[j] > 0.15 else edge, rotation=90,
                            va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in order], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11, pad=8)
    if show_ylabel:
        ax.set_ylabel("Unfaithfulness rate", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    vindr = "results/faith_vindr_all_fixed.jsonl"
    nih = "results/faith_nih_all_fixed.jsonl"

    for p in (vindr, nih):
        if not os.path.exists(p):
            print(f"[fig2] 파일 없음: {p}")
            return

    sv = load_stats(vindr)
    sn = load_stats(nih)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    draw_panel(axes[0], sv, "VinDr-CXR", True)
    draw_panel(axes[1], sn, "NIH ChestX-ray14", False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = "results/fig2_unfaithfulness_both.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[fig2] 저장: {out}")

    print("\n확인용 수치:")
    for name, st in [("VinDr", sv), ("NIH", sn)]:
        print(f"\n[{name}]")
        for k in sorted(st):
            n, u, rate, lo, hi = st[k]
            print(f"  {k[0]:12} {k[2]:8} {k[1]:15} "
                  f"n={n:4} rate={rate:.2f} [{lo:.2f},{hi:.2f}]")


if __name__ == "__main__":
    main()