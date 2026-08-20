#!/usr/bin/env python
"""
스크립트 08: 여러 모델의 결과 파일 병합 (중복 제거).

사용:
    python scripts/08_merge_results.py --kind explanations \
        --inputs a.jsonl b.jsonl c.jsonl --output data/cache/explanations.jsonl
    python scripts/08_merge_results.py --kind faith \
        --inputs g.jsonl m.jsonl c.jsonl --output results/faith_results.jsonl
"""
import argparse
import json
import os


def key_expl(d):
    return f"{d['case_id']}__{d['model_id']}__{d['prompt_condition']}"


def key_faith(d):
    det = json.dumps(d.get("detail", {}), sort_keys=True)
    return (f"{d['case_id']}__{d['model_id']}__{d['prompt_condition']}"
            f"__{d['test_name']}__{det}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["explanations", "faith"], required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    keyfn = key_expl if args.kind == "explanations" else key_faith

    merged = {}
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[merge] 건너뜀(없음): {path}")
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                merged[keyfn(d)] = d
                n += 1
        print(f"[merge] {path}: {n}건 읽음")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for d in merged.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    from collections import Counter
    by_model = Counter(d["model_id"] for d in merged.values())
    print(f"\n[merge] 병합 완료: 총 {len(merged)}건 -> {args.output}")
    print("[merge] 모델별:")
    for m, c in by_model.most_common():
        print(f"    {m}: {c}")


if __name__ == "__main__":
    main()