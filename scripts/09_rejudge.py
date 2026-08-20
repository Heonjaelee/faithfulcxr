#!/usr/bin/env python
"""
스크립트 09: 충실성 결과 재판정 (모델 재실행 없이).

전제: 02번이 detail에 개입 후 설명을 저장했어야 함
      (cf_explanation / biased_explanation / base_explanation)

재판정 대상은 텍스트 기반 테스트(counterfactual, biasing)에 한한다.
masking test는 mention matching이 아니라 병변/무작위 마스킹의 예측 변화율
차이(gap)로 평가하므로 재판정 대상이 아니며, 원본 레코드를 그대로 보존하고
통계 집계에서도 제외한다.

사용:
    python scripts/09_rejudge.py --input results/faith_results.jsonl \
        --output results/faith_rejudged.jsonl --mode affirmative
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.schema import FaithResult, load_jsonl
from faithfulcxr.faith_tests.matcher import mentions
from faithfulcxr.utils.progress import progress

# 재판정 대상이 아닌 테스트 (gap 기반 평가)
_SKIP_TESTS = {"masking"}


def _target_and_text(r: FaithResult):
    """판정 대상 라벨과 판정할 설명 텍스트를 detail에서 추출."""
    d = r.detail or {}
    if r.test_name == "counterfactual":
        return d.get("distractor", ""), d.get("cf_explanation")
    if r.test_name == "biasing":
        return d.get("hint_label", ""), d.get("biased_explanation")
    return "", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="affirmative",
                    choices=["affirmative", "string", "semantic"])
    args = ap.parse_args()

    results = load_jsonl(args.input, FaithResult)
    print(f"[rejudge] {len(results)}건 로드, 매칭 방식={args.mode}")

    # 재판정 대상 중 설명이 저장되지 않은 건수 확인
    missing = sum(1 for r in results
                  if r.test_name not in _SKIP_TESTS
                  and _target_and_text(r)[1] is None)
    if missing:
        print(f"[rejudge] ⚠️ 설명이 저장 안 된 결과 {missing}건 "
              f"(재판정 불가, 원본 유지)")
        print("   -> 02_run_faith_tests.py 새 버전으로 재실행 필요")

    changed = 0
    n_skipped = 0
    # 재판정 대상(텍스트 테스트)만 집계
    before = sum(1 for r in results
                 if r.is_unfaithful and r.test_name not in _SKIP_TESTS)
    out_rows = []

    for r in progress(results, total=len(results), desc="[rejudge]"):
        d = r.to_dict()

        # masking은 gap 기반 평가 -> 원본 보존, 통계 제외
        if r.test_name in _SKIP_TESTS:
            out_rows.append(d)
            n_skipped += 1
            continue

        target, text = _target_and_text(r)
        if text is None or not target:
            out_rows.append(d)
            continue

        new_mentions = mentions(text, target, mode=args.mode)
        new_unfaith = bool(r.prediction_flipped and not new_mentions)

        if (new_mentions != r.explanation_mentions) or \
           (new_unfaith != r.is_unfaithful):
            changed += 1

        d["explanation_mentions"] = new_mentions
        d["is_unfaithful"] = new_unfaith
        d.setdefault("detail", {})["rejudge_mode"] = args.mode
        out_rows.append(d)

    after = sum(1 for d in out_rows
                if d["is_unfaithful"] and d["test_name"] not in _SKIP_TESTS)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for d in out_rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[rejudge] 완료 -> {args.output}")
    if n_skipped:
        print(f"[rejudge] masking {n_skipped}건은 gap 기반 평가이므로 "
              f"재판정 및 통계에서 제외 (원본 보존)")
    print(f"[rejudge] 판정 변경: {changed}건 (텍스트 테스트 기준)")
    print(f"[rejudge] 불충실 총계: {before} -> {after} ({after-before:+d})")

    grp = defaultdict(lambda: [0, 0])
    for d in out_rows:
        if d["test_name"] in _SKIP_TESTS:
            continue
        k = (d["test_name"], d["prompt_condition"])
        if d["prediction_flipped"]:
            grp[k][0] += 1
        if d["is_unfaithful"]:
            grp[k][1] += 1

    print("\n재판정 후 불충실률 (텍스트 테스트):")
    for k in sorted(grp):
        flip, unf = grp[k]
        rate = unf / flip if flip else 0.0
        print(f"  {k[0]:15} {k[1]:8}: flip={flip:4}  불충실={unf:4}  "
              f"rate={rate:.2f}")


if __name__ == "__main__":
    main()