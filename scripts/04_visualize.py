#!/usr/bin/env python
"""
스크립트 04: 결과 시각화.

results/faith_results.jsonl 을 읽어 논문/발표용 그림을 생성합니다:

  fig1_unfaithfulness_by_condition.png
      테스트 × 조건별 불충실률 막대그래프 + 부트스트랩 95% CI 에러바.
  fig2_flip_vs_mention.png
      조건별 (예측 flip 수) vs (설명 언급 수) 비교.

사용:
    python scripts/04_visualize.py
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.utils.io import load_config
from faithfulcxr.schema import FaithResult, load_jsonl
from faithfulcxr.analysis.metrics import unfaithfulness_rate, bootstrap_ci


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
    print(f"[viz] {len(results)}건 로드")

    out_dir = cfg["paths"]["results"]

    groups = defaultdict(list)
    for r in results:
        groups[(r.test_name, r.prompt_condition)].append(r)

    # ---- Figure 1: 불충실률 + CI ----
    tests = sorted(set(t for t, _ in groups))
    conds = ["cot", "posthoc"]
    colors = {"cot": "#d1495b", "posthoc": "#3d5a80"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(tests))
    width = 0.36
    for i, cond in enumerate(conds):
        rates, errs_lo, errs_hi = [], [], []
        for test in tests:
            rs = groups.get((test, cond), [])
            stat = unfaithfulness_rate(rs)
            rate = stat["unfaithfulness_rate"]
            rate = 0 if rate != rate else rate
            lo, hi = bootstrap_ci(rs, n_boot=cfg["analysis"]["bootstrap_n"])
            lo = rate if lo != lo else lo
            hi = rate if hi != hi else hi
            rates.append(rate)
            errs_lo.append(max(0, rate - lo))
            errs_hi.append(max(0, hi - rate))
        offs = [xi + (i - 0.5) * width for xi in x]
        ax.bar(offs, rates, width, label=cond, color=colors[cond],
               yerr=[errs_lo, errs_hi], capsize=4, alpha=0.9)
        for xo, rt in zip(offs, rates):
            ax.text(xo, rt + 0.02, f"{rt:.2f}", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tests)
    ax.set_ylabel("Unfaithfulness rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unfaithfulness by test and prompt condition\n"
                 "(error bars = bootstrap 95% CI)")
    ax.legend(title="condition")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(out_dir, "fig1_unfaithfulness_by_condition.png")
    fig.savefig(f1, dpi=150)
    print(f"[viz] 저장: {f1}")

    # ---- Figure 2: flip vs mention ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    labels, flips, mentions = [], [], []
    for test in tests:
        for cond in conds:
            rs = groups.get((test, cond), [])
            if not rs:
                continue
            labels.append(f"{test}\n{cond}")
            flips.append(sum(1 for r in rs if r.prediction_flipped))
            mentions.append(sum(1 for r in rs if r.explanation_mentions))
    xi = range(len(labels))
    width = 0.4
    ax2.bar([i - width/2 for i in xi], flips, width,
            label="prediction flips", color="#ee6c4d")
    ax2.bar([i + width/2 for i in xi], mentions, width,
            label="explanation mentions", color="#98c1d9")
    ax2.set_xticks(list(xi))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("count")
    ax2.set_title("Prediction flips vs. explanation mentions\n"
                  "(gap suggests unfaithfulness)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    f2 = os.path.join(out_dir, "fig2_flip_vs_mention.png")
    fig2.savefig(f2, dpi=150)
    print(f"[viz] 저장: {f2}")

    print("\n두 그림을 results/ 폴더에서 확인하세요.")


if __name__ == "__main__":
    main()