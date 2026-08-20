#!/usr/bin/env python
"""
스크립트 07: 마스킹 테스트 전용 분석.

핵심 질문: 병변 영역을 가릴 때, 무작위 영역보다 예측이 '더 많이' 바뀌는가?
    lesion_flip_rate > random_flip_rate => 시각적 근거가 예측을 좌우 (충실)

사용:
    python scripts/07_masking_analysis.py
"""
import argparse
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

    path = os.path.join(cfg["paths"]["results"], "faith_results.jsonl")
    results = load_jsonl(path, FaithResult)
    masking = [r for r in results if r.test_name == "masking"]

    if not masking:
        print("[mask] 마스킹 결과가 없습니다.")
        sys.exit(1)
    print(f"[mask] 마스킹 결과 {len(masking)}건")

    by_cond = defaultdict(lambda: {"lesion": [], "random": []})
    for r in masking:
        lf = r.detail.get("lesion_flip")
        rf = r.detail.get("random_flip")
        if lf is not None:
            by_cond[r.prompt_condition]["lesion"].append(1 if lf else 0)
        if rf is not None:
            by_cond[r.prompt_condition]["random"].append(1 if rf else 0)

    conds = sorted(by_cond)
    lesion_rates, random_rates = [], []
    print("\n조건별 flip rate (병변 vs 무작위):")
    for c in conds:
        lr = sum(by_cond[c]["lesion"]) / len(by_cond[c]["lesion"]) if by_cond[c]["lesion"] else 0
        rr = sum(by_cond[c]["random"]) / len(by_cond[c]["random"]) if by_cond[c]["random"] else 0
        lesion_rates.append(lr)
        random_rates.append(rr)
        gap = lr - rr
        verdict = "충실 신호(병변>무작위)" if gap > 0.05 else \
                  "약함/역전(불충실 신호)" if gap < -0.05 else "차이 미미"
        print(f"  {c:8}: 병변={lr:.2f}  무작위={rr:.2f}  차이={gap:+.2f}  [{verdict}]")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(conds))
    width = 0.36
    ax.bar([i - width/2 for i in x], lesion_rates, width,
           label="lesion masking", color="#e07a5f")
    ax.bar([i + width/2 for i in x], random_rates, width,
           label="random masking (control)", color="#81b29a")
    for i, (lr, rr) in enumerate(zip(lesion_rates, random_rates)):
        ax.text(i - width/2, lr + 0.02, f"{lr:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, rr + 0.02, f"{rr:.2f}", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(conds)
    ax.set_ylabel("Prediction flip rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Masking test: lesion vs. random region\n"
                 "(lesion > random = model relies on the visual evidence)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(cfg["paths"]["results"], "fig4_masking.png")
    fig.savefig(out, dpi=150)
    print(f"\n[mask] 저장: {out}")


if __name__ == "__main__":
    main()