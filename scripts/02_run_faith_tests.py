#!/usr/bin/env python
"""
스크립트 2: 충실성 테스트 실행 (재개 가능 / resumable).

★ 재개 기능:
    - 시작 시 기존 결과를 로드해, 이미 끝난 (모델×케이스×조건×테스트) 단위를 건너뜀.
    - 각 단위가 끝날 때마다 파일에 append 저장 -> 끊겨도 직전까지 보존.

사용:
    python scripts/02_run_faith_tests.py --limit 20 --tests counterfactual biasing
    python scripts/02_run_faith_tests.py --limit 20 --restart   # 처음부터 다시
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.utils.io import load_config, set_seed, ensure_dirs
from faithfulcxr.data.loader import load_cases
from faithfulcxr.models.base import get_backend
from faithfulcxr.schema import FaithResult, load_jsonl
from faithfulcxr.faith_tests.text_tests import counterfactual_test, biasing_test
from faithfulcxr.faith_tests.masking_test import masking_test


def _append_jsonl(results, path):
    with open(path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def _done_units(path):
    """이미 완료된 (모델, 케이스, 조건, 테스트) 단위 집합."""
    done = set()
    if not os.path.exists(path):
        return done
    for r in load_jsonl(path, FaithResult):
        done.add((r.model_id, r.case_id, r.prompt_condition, r.test_name))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tests", nargs="+",
                    default=["counterfactual", "biasing", "masking"])
    ap.add_argument("--restart", action="store_true",
                    help="기존 결과를 지우고 처음부터")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    cases = load_cases(cfg)
    if args.limit:
        cases = cases[:args.limit]

    conditions = cfg["prompting"]["conditions"]
    models = [m for m in cfg["models"] if m.get("enabled", True)]
    ft = cfg["faith_tests"]
    mask_cache = os.path.join(cfg["paths"]["cache"], "masked_images")
    out = os.path.join(cfg["paths"]["results"], "faith_results.jsonl")

    from faithfulcxr.utils.progress import progress

    if args.restart and os.path.exists(out):
        os.remove(out)
        print("[test] --restart: 기존 결과 삭제, 처음부터 시작")

    done = _done_units(out)
    if done:
        print(f"[test] 완료된 단위 {len(done)}개 발견 -> 건너뜀")

    plan = []
    for mcfg in models:
        for ci, case in enumerate(cases):
            for cond in conditions:
                todo_tests = [t for t in args.tests
                              if (mcfg["id"], case.case_id, cond, t) not in done]
                if todo_tests:
                    plan.append((mcfg, ci, case, cond, todo_tests))

    total_units = len(plan)
    if total_units == 0:
        print("[test] 모든 단위가 이미 완료됨. 할 일 없음.")
        print(f"[test] 결과: {out}")
        return

    print(f"[test] 남은 단위 {total_units}개 처리 시작")

    backends = {}
    processed = 0
    for mcfg, ci, case, cond, todo_tests in progress(
            plan, total=total_units, desc="[test]"):
        if mcfg["id"] not in backends:
            backends[mcfg["id"]] = get_backend(mcfg, force_mock=args.mock)
        backend = backends[mcfg["id"]]
        seed = cfg["project"]["seed"] + ci

        unit_results = []
        if "counterfactual" in todo_tests:
            unit_results += counterfactual_test(
                case, backend, cond,
                n_interventions=ft["counterfactual"]["n_interventions_per_case"],
                seed=seed)
        if "biasing" in todo_tests:
            unit_results += biasing_test(
                case, backend, cond,
                hint_templates=ft["biasing"]["hint_templates"],
                seed=seed)
        if "masking" in todo_tests:
            unit_results += masking_test(
                case, backend, cond, cache_dir=mask_cache,
                fill_method=ft["masking"]["fill_method"],
                blur_kernel=ft["masking"]["blur_kernel"],
                random_control=ft["masking"]["random_control"],
                seed=seed)

        if unit_results:
            _append_jsonl(unit_results, out)
        processed += 1

    print(f"[test] 완료. 이번 실행 {processed}개 단위 처리 -> {out}")
    total = len(load_jsonl(out, FaithResult))
    print(f"[test] 결과 파일 총 {total}건")


if __name__ == "__main__":
    main()