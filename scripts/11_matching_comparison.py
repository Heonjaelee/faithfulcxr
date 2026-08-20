#!/usr/bin/env python
"""
스크립트 11: 매칭 방식 비교 그림 (Figure 5).

string matching과 affirmative matching의 unfaithfulness rate를
조건별로 나란히 비교한다. 두 재판정 결과 파일을 입력으로 받는다.

사용:
    python scripts/09_rejudge.py --input results_colab/faith_gpt_vindr.jsonl \
        --output results/tmp_gpt_string.jsonl --mode string
    python scripts/09_rejudge.py --input results_colab/faith_medgemma_vindr_clean.jsonl \
        --output results/tmp_med_string.jsonl --mode string
    python scripts/11_matching_comparison.py
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rates(path, model_id=None):
    """조건별 unfaithfulness rate 산출 (텍스트 테스트만)."""
    grp = defaultdict(lambda: [0, 0])
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["test_name"] == "masking":
                continue
            if model_id and r["model_id"] != model_id:
                continue
            k = (r["test_name"], r["prompt_condition"])
            if r["prediction_flipped"]:
                grp[k][0] += 1
            if r["is_unfaithful"]:
                grp[k][1] += 1
    return {k: (u / f if f else 0.0) for k, (f, u) in grp.items()}


def main():
    # 입력 파일
    aff_path = "results/faith_vindr_all_fixed.jsonl"
    gpt_str = "results/tmp_gpt_string.jsonl"
    med_str = "results/tmp_med_string.jsonl"

    for p in (aff_path, gpt_str, med_str):
        if not os.path.exists(p):
            print(f"[fig5] 파일 없음: {p}")
            print("   -> 09_rejudge.py를 먼저 실행하세요 (--mode string)")
            return

    aff_gpt = rates(aff_path, "gpt54")
    aff_med = rates(aff_path, "medgemma4b")
    str_gpt = rates(gpt_str)
    str_med = rates(med_str)

    # 표시 순서: (모델, 조건, 테스트)
    rows = [
        ("GPT-5.4", "posthoc", "counterfactual", str_gpt, aff_gpt),
        ("GPT-5.4", "cot", "counterfactual", str_gpt, aff_gpt),
        ("GPT-5.4", "posthoc", "biasing", str_gpt, aff_gpt),
        ("GPT-5.4", "cot", "biasing", str_gpt, aff_gpt),
        ("MedGemma-4B", "posthoc", "counterfactual", str_med, aff_med),
        ("MedGemma-4B", "cot", "counterfactual", str_med, aff_med),
        ("MedGemma-4B", "posthoc", "biasing", str_med, aff_med),
        ("MedGemma-4B", "cot", "biasing", str_med, aff_med),
    ]

    labels, s_vals, a_vals = [], [], []
    for model, cond, test, s_src, a_src in rows:
        key = (test, cond)
        s_vals.append(s_src.get(key, 0.0))
        a_vals.append(a_src.get(key, 0.0))
        short_test = "CF" if test == "counterfactual" else "Bias"
        short_cond = "CoT" if cond == "cot" else "Post"
        labels.append(f"{short_test}\n{short_cond}")

    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - width/2, s_vals, width, label="String matching",
                color="#B4B2A9", edgecolor="#5F5E5A", linewidth=0.6)
    b2 = ax.bar(x + width/2, a_vals, width, label="Affirmative matching",
                color="#7F77DD", edgecolor="#3C3489", linewidth=0.6)

    # 증가폭 표시
    for i, (s, a) in enumerate(zip(s_vals, a_vals)):
        d = a - s
        if d > 0.005:
            ax.annotate(f"+{d:.2f}", xy=(x[i] + width/2, a),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color="#3C3489")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Unfaithfulness rate", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # 모델 구분선과 라벨
    ax.axvline(3.5, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax.text(1.5, 1.02, "GPT-5.4", ha="center", fontsize=10)
    ax.text(5.5, 1.02, "MedGemma-4B", ha="center", fontsize=10)

    plt.tight_layout()
    out = "results/fig5_matching_comparison.png"
    plt.savefig(out, dpi=200)
    print(f"[fig5] 저장: {out}")

    print("\n확인용 수치:")
    for (model, cond, test, _, _), s, a in zip(rows, s_vals, a_vals):
        print(f"  {model:12} {cond:8} {test:15} "
              f"string={s:.2f} affirmative={a:.2f} Δ={a-s:+.2f}")


if __name__ == "__main__":
    main()