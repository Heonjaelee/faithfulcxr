#!/usr/bin/env python
"""
스크립트 05: LLM-judge 그럴듯함 평가.

data/cache/explanations.jsonl 의 각 설명에 그럴듯함(1~5)을 매겨
results/plausibility.jsonl 에 저장. 이미 평가된 건 건너뜀.

사용:
    python scripts/05_judge_plausibility.py --limit 10
    python scripts/05_judge_plausibility.py --mock   # 파이프라인 검증
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.analysis.judge import score_plausibility, make_judge
from faithfulcxr.utils.io import load_config, set_seed
from faithfulcxr.schema import Explanation, load_jsonl
from faithfulcxr.analysis.judge import score_plausibility, make_openai_judge
from faithfulcxr.utils.progress import progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--input", default=None,
                    help="설명 파일 경로 (기본: data/cache/explanations_medgemma.jsonl). "
                         "모델별 파일을 지정하면 그것만 평가합니다.")
    ap.add_argument("--output", default=None,
                    help="결과 저장 경로 (기본: results/plausibility.jsonl)")
    ap.add_argument("--judge", default="gemini",
                    choices=["gemini", "anthropic", "openai"],
                    help="judge 제공자 (편향 회피로 gemini/anthropic 권장)")
    ap.add_argument("--judge-model", default=None,
                    help="judge 모델명 (미지정 시 기본값)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    # 입력 경로: --input이 있으면 그것, 없으면 기본 캐시
    expl_path = args.input or os.path.join(cfg["paths"]["cache"],
                                           "explanations.jsonl")
    if not os.path.exists(expl_path):
        print(f"[judge] 설명 파일이 없습니다: {expl_path}")
        sys.exit(1)

    expls = load_jsonl(expl_path, Explanation)
    if args.limit:
        expls = expls[:args.limit]
    print(f"[judge] 평가 대상 설명 {len(expls)}건")

    # 출력 경로: --output이 있으면 그것, 없으면 기본
    out_path = args.output or os.path.join(cfg["paths"]["results"],
                                           "plausibility.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done[d["key"]] = d
        print(f"[judge] 기존 평가 {len(done)}건 로드")

    n_repeats = cfg["judge"]["n_repeats"]
    if args.mock:
        import random
        rng = random.Random(0)
        def judge_fn(prompt):
            return str(rng.randint(3, 5))
    else:
        judge_fn = make_judge(args.judge, args.judge_model)
        print(f"[judge] 사용 judge: {args.judge} "
              f"({args.judge_model or '기본 모델'})")

    todo = [e for e in expls if e.key() not in done]
    print(f"[judge] 신규 평가 {len(todo)}건 (n_repeats={n_repeats})")

    results = list(done.values())
    n_new = 0
    for e in progress(todo, total=len(todo), desc="[judge] 그럴듯함"):
        score = score_plausibility(e.explanation_text, judge_fn, n_repeats)
        rec = {
            "key": e.key(), "case_id": e.case_id, "model_id": e.model_id,
            "prompt_condition": e.prompt_condition,
            "prediction": e.prediction, "plausibility": score,
        }
        results.append(rec)
        done[e.key()] = rec
        n_new += 1
        if n_new % 10 == 0:
            _dump(results, out_path)

    _dump(results, out_path)
    print(f"[judge] 완료. 신규 {n_new}건, 총 {len(results)}건 -> {out_path}")

    valid = [r["plausibility"] for r in results
             if r["plausibility"] == r["plausibility"]]
    if valid:
        print(f"[judge] 평균 그럴듯함: {sum(valid)/len(valid):.2f} "
              f"(min {min(valid):.1f}, max {max(valid):.1f})")


def _dump(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()