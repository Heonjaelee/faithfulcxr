#!/usr/bin/env python
"""
스크립트 06b: 모델별 사분면 개수 확인 (06_quadrant.py와 동일 로직, model_id로 분리 집계)

사용:
    python scripts/06b_quadrant_by_model.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.utils.io import load_config
from faithfulcxr.schema import FaithResult, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    res_dir = cfg["paths"]["results"]
    plaus_path = os.path.join(res_dir, "plausibility.jsonl")
    faith_path = os.path.join(res_dir, "faith_results.jsonl")

    if not os.path.exists(plaus_path):
        print(f"[quad-by-model] 그럴듯함 결과 없음: {plaus_path}")
        sys.exit(1)

    # plausibility.jsonl 로드 (06_quadrant.py와 동일)
    plaus = {}
    with open(plaus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                plaus[(d["case_id"], d["model_id"], d["prompt_condition"])] = d["plausibility"]

    # faith_results.jsonl 로드 (06_quadrant.py와 동일)
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

    # model_id 별로 교차 포인트 모으기
    by_model = defaultdict(list)  # model_id -> list of (x, y, condition)
    for key, p in plaus.items():
        if key in faith_score and p == p:
            case_id, model_id, cond = key
            by_model[model_id].append((p, faith_score[key], cond))

    if not by_model:
        print("[quad-by-model] 교차 가능한 데이터 포인트가 없습니다.")
        sys.exit(1)

    x_mid, y_mid = 3.0, 0.5

    print(f"\n=== model_id 종류: {sorted(by_model.keys())} ===\n")

    for model_id, pts in sorted(by_model.items()):
        n_total = len(pts)
        n_pu = sum(1 for x, y, c in pts if x >= x_mid and y < y_mid)   # 그럴듯 & 불충실
        n_pf = sum(1 for x, y, c in pts if x >= x_mid and y >= y_mid)  # 그럴듯 & 충실
        n_iu = sum(1 for x, y, c in pts if x < x_mid and y < y_mid)    # 안그럴듯 & 불충실
        n_if = sum(1 for x, y, c in pts if x < x_mid and y >= y_mid)   # 안그럴듯 & 충실
        avg_x = sum(x for x, y, c in pts) / n_total
        avg_y = sum(y for x, y, c in pts) / n_total

        print(f"--- {model_id} (총 {n_total}개 포인트) ---")
        print(f"  평균 plausibility: {avg_x:.2f} / 평균 faithfulness: {avg_y:.2f}")
        print(f"  그럴듯 & 충실:     {n_pf}  ({100*n_pf/n_total:.0f}%)")
        print(f"  그럴듯 & 불충실:   {n_pu}  ({100*n_pu/n_total:.0f}%)  <- RQ2 핵심")
        print(f"  안그럴듯 & 충실:   {n_if}  ({100*n_if/n_total:.0f}%)")
        print(f"  안그럴듯 & 불충실: {n_iu}  ({100*n_iu/n_total:.0f}%)")
        print()


if __name__ == "__main__":
    main()