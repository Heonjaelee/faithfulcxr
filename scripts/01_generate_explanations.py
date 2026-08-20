#!/usr/bin/env python
"""
스크립트 1: 설명 생성 + 캐싱 (진행 표시 포함).

케이스 풀 × (활성화된 모델) × (프롬프트 조건)으로 설명을 생성하고
data/cache/explanations.jsonl 에 저장합니다.
이미 캐시에 있는 (case × model × condition)은 건너뜁니다.

사용:
    python scripts/01_generate_explanations.py
    python scripts/01_generate_explanations.py --mock
    python scripts/01_generate_explanations.py --limit 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.utils.io import load_config, set_seed, ensure_dirs
from faithfulcxr.data.loader import load_cases
from faithfulcxr.models.base import get_backend
from faithfulcxr.schema import Explanation, dump_jsonl, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mock", action="store_true", help="모든 모델 강제 mock")
    ap.add_argument("--limit", type=int, default=None, help="케이스 수 제한(파일럿)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    cases = load_cases(cfg)
    if args.limit:
        cases = cases[:args.limit]
        print(f"[gen] 파일럿 모드: {len(cases)}케이스로 제한")

    cache_path = os.path.join(cfg["paths"]["cache"], "explanations.jsonl")
    existing = {}
    if os.path.exists(cache_path):
        for e in load_jsonl(cache_path, Explanation):
            existing[e.key()] = e
        print(f"[gen] 기존 캐시 {len(existing)}건 로드")

    conditions = cfg["prompting"]["conditions"]
    models = [m for m in cfg["models"] if m.get("enabled", True)]

    from faithfulcxr.utils.progress import progress

    import json
    all_expls = list(existing.values())
    n_new = 0

    def _append(expl):
        # 매 건 즉시 append 저장 -> 세션이 언제 끊겨도 직전까지 보존
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(expl.to_dict(), ensure_ascii=False) + "\n")

    for mcfg in models:
        backend = get_backend(mcfg, force_mock=args.mock)
        todo = [(case, cond) for case in cases for cond in conditions
                if f"{case.case_id}__{mcfg['id']}__{cond}" not in existing]
        if not todo:
            print(f"[gen] {mcfg['display']}: 이미 모두 캐시됨, 건너뜀")
            continue

        desc = f"[gen] {mcfg['display']}"
        for case, cond in progress(todo, total=len(todo), desc=desc):
            expl = backend.predict(case.image_path, cond)
            expl.case_id = case.case_id
            all_expls.append(expl)
            existing[f"{case.case_id}__{mcfg['id']}__{cond}"] = expl
            _append(expl)          # 즉시 저장
            n_new += 1

    print(f"[gen] 완료. 신규 {n_new}건, 총 {len(all_expls)}건 -> {cache_path}")


if __name__ == "__main__":
    main()