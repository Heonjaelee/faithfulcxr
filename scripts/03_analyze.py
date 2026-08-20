#!/usr/bin/env python
"""
스크립트 3: 분석 + 리포트.

results/faith_results.jsonl 을 읽어 (model × condition × test)별 불충실 지표를
표로 만들고, results/summary.csv 와 콘솔 요약을 출력합니다.

사용:
    python scripts/03_analyze.py
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.utils.io import load_config
from faithfulcxr.schema import FaithResult, load_jsonl
from faithfulcxr.analysis.metrics import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    path = os.path.join(cfg["paths"]["results"], "faith_results.jsonl")
    results = load_jsonl(path, FaithResult)
    print(f"[analyze] {len(results)}건 로드")

    rows = summarize(results, n_boot=cfg["analysis"]["bootstrap_n"])

    # CSV 저장 (행마다 키가 다를 수 있으므로 합집합으로 헤더 구성)
    out = os.path.join(cfg["paths"]["results"], "summary.csv")
    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"[analyze] 요약 -> {out}\n")

    # 콘솔 표
    print(f"{'model':<12}{'cond':<9}{'test':<15}"
          f"{'n_flip':>7}{'unfaith':>9}{'rate':>7}{'  95% CI':>16}")
    print("-" * 75)
    for r in rows:
        ci = f"[{r['ci_low']:.2f},{r['ci_high']:.2f}]" \
            if r['ci_low'] == r['ci_low'] else "[nan]"
        rate = r["unfaithfulness_rate"]
        rate_s = f"{rate:.2f}" if rate == rate else "nan"
        print(f"{r['model']:<12}{r['condition']:<9}{r['test']:<15}"
              f"{r['n_flipped']:>7}{r['n_unfaithful']:>9}{rate_s:>7}{ci:>16}")
        if r["test"] == "masking" and "faithful_gap" in r:
            print(f"    └ masking: lesion_flip={r['lesion_flip_rate']:.2f} "
                  f"random_flip={r['random_flip_rate']:.2f} "
                  f"gap={r['faithful_gap']:+.2f} (양수여야 충실)")


if __name__ == "__main__":
    main()
