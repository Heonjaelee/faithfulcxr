#!/usr/bin/env python
"""
스크립트 06: 그럴듯함 × 충실성 사분면 분석 (RQ2 핵심).

x축 = 그럴듯함 (plausibility.jsonl, 1~5)
y축 = 충실성   (faith_results.jsonl, case×condition별 1 - 불충실률)
오른쪽 아래(그럴듯한데 불충실)에 점이 많으면 RQ2 가설 증명.

사용:
    python scripts/06_quadrant.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.utils.io import load_config
from faithfulcxr.schema import FaithResult, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res_dir = cfg["paths"]["results"]
    plaus_path = os.path.join(res_dir, "plausibility.jsonl")
    faith_path = os.path.join(res_dir, "faith_results.jsonl")

    if not os.path.exists(plaus_path):
        print(f"[quad] 그럴듯함 결과 없음: {plaus_path}")
        sys.exit(1)

    plaus = {}
    with open(plaus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                plaus[(d["case_id"], d["model_id"], d["prompt_condition"])] = d["plausibility"]

    faith = load_jsonl(faith_path, FaithResult)
    grp = defaultdict(list)
    for r in faith:
        grp[(r.case_id, r.model_id, r.prompt_condition)].append(r)

    faith_score = {}
    for key, rs in grp.items():
        flipped = [r for r in rs if r.prediction_flipped]
        if not flipped:
            continue
        unfaith = sum(1 for r in flipped if r.is_unfaithful) / len(flipped)
        faith_score[key] = 1.0 - unfaith

    xs, ys, colors_pt, markers_pt = [], [], [], []
    cond_color = {"cot": "#d1495b", "posthoc": "#3d5a80"}
    model_marker = {"gpt54": "o", "medgemma4b": "^"}
    for key, p in plaus.items():
        if key in faith_score and p == p:
            case_id, model_id, cond = key
            xs.append(p)
            ys.append(faith_score[key])
            colors_pt.append(cond_color.get(cond, "#888"))
            markers_pt.append(model_marker.get(model_id, "o"))

    if not xs:
        print("[quad] 교차 가능한 데이터 포인트가 없습니다.")
        sys.exit(1)
    print(f"[quad] {len(xs)}개 데이터 포인트")

    fig, ax = plt.subplots(figsize=(7, 7))
    # 마커 모양(model)별로 나눠서 scatter — matplotlib은 marker를 배열로 못 받으므로 그룹화
    for marker in set(markers_pt):
        idx = [i for i, m in enumerate(markers_pt) if m == marker]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx],
                   c=[colors_pt[i] for i in idx], marker=marker,
                   alpha=0.6, s=60, edgecolors="white", linewidths=0.5)

    x_mid, y_mid = 3.0, 0.5
    ax.axvline(x_mid, color="gray", ls="--", alpha=0.5)
    ax.axhline(y_mid, color="gray", ls="--", alpha=0.5)

    ax.text(4.5, 0.9, "plausible\n& faithful", ha="center", color="green", alpha=0.7)
    ax.text(4.5, 0.1, "plausible\nbut UNFAITHFUL", ha="center", color="red",
            fontweight="bold", alpha=0.8)
    ax.text(1.7, 0.9, "implausible\nbut faithful", ha="center", color="gray", alpha=0.7)
    ax.text(1.7, 0.1, "implausible\n& unfaithful", ha="center", color="gray", alpha=0.7)

    ax.set_xlabel("Plausibility (LLM-judge, 1-5)")
    ax.set_ylabel("Faithfulness (1 - unfaithfulness rate)")
    ax.set_xlim(1, 5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Plausibility vs. Faithfulness\n"
                 "(bottom-right = plausible but unfaithful = the key finding)")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(color="#d1495b", label="cot"),
        Patch(color="#3d5a80", label="posthoc"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=9, label="gpt54"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=9, label="medgemma4b"),
    ]
    ax.legend(handles=handles, title="condition / model", loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out = os.path.join(res_dir, "fig3_quadrant.png")
    fig.savefig(out, dpi=150)
    print(f"[quad] 저장: {out}")

    n_pu = sum(1 for x, y in zip(xs, ys) if x >= x_mid and y < y_mid)
    n_pf = sum(1 for x, y in zip(xs, ys) if x >= x_mid and y >= y_mid)
    print(f"\n사분면 분포 (전체):")
    print(f"  그럴듯 & 충실:   {n_pf}")
    print(f"  그럴듯 & 불충실: {n_pu}  <- RQ2 핵심")

    # 모델별 사분면 분포도 함께 출력
    model_of = {}
    for key, p in plaus.items():
        if key in faith_score and p == p:
            model_of[(p, faith_score[key])] = key[1]
    by_model_counts = defaultdict(lambda: defaultdict(int))
    for x, y in zip(xs, ys):
        m = model_of.get((x, y), "?")
        quad = ("plausible" if x >= x_mid else "implausible") + "_" + \
               ("faithful" if y >= y_mid else "unfaithful")
        by_model_counts[m][quad] += 1
    print(f"\n사분면 분포 (모델별):")
    for m, counts in sorted(by_model_counts.items()):
        print(f"  {m}: {dict(counts)}")


if __name__ == "__main__":
    main()