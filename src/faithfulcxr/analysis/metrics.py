"""
충실성 지표 집계.

    unfaithfulness_rate  = N_unfaithful / N_flipped   (Atanasova 공식)
                           예측이 바뀐 케이스 중 설명이 언급 안 한 비율.

    bootstrap_ci         비율의 부트스트랩 신뢰구간 (표본이 작을 때 필수).

    masking 전용         lesion_flip_rate vs random_flip_rate 비교
                         (병변 마스킹이 무작위보다 더 flip시켜야 충실).

결과를 (모델 × 조건 × 테스트)로 groupby하여 표로 만듭니다.
"""
from __future__ import annotations

import random
from collections import defaultdict

from faithfulcxr.schema import FaithResult


def unfaithfulness_rate(results: list[FaithResult]) -> dict:
    """flip된 케이스 기준 불충실 비율."""
    flipped = [r for r in results if r.prediction_flipped]
    n_flip = len(flipped)
    n_unfaith = sum(1 for r in flipped if r.is_unfaithful)
    rate = (n_unfaith / n_flip) if n_flip else float("nan")
    return {"n_total": len(results), "n_flipped": n_flip,
            "n_unfaithful": n_unfaith, "unfaithfulness_rate": rate}


def bootstrap_ci(results: list[FaithResult], n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """불충실 비율의 95% 부트스트랩 CI."""
    flipped = [1 if r.is_unfaithful else 0
               for r in results if r.prediction_flipped]
    if not flipped:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = []
    n = len(flipped)
    for _ in range(n_boot):
        sample = [flipped[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return (lo, hi)


def masking_control_stats(results: list[FaithResult]) -> dict:
    """masking 테스트: 병변 vs 무작위 flip rate."""
    mask = [r for r in results if r.test_name == "masking"]
    if not mask:
        return {}
    lesion = [r.detail.get("lesion_flip") for r in mask]
    rand = [r.detail.get("random_flip") for r in mask
            if r.detail.get("random_flip") is not None]
    lesion_rate = sum(1 for x in lesion if x) / len(lesion) if lesion else float("nan")
    rand_rate = sum(1 for x in rand if x) / len(rand) if rand else float("nan")
    return {"lesion_flip_rate": lesion_rate, "random_flip_rate": rand_rate,
            "faithful_gap": lesion_rate - rand_rate}


def summarize(results: list[FaithResult], n_boot: int = 2000) -> list[dict]:
    """(model, condition, test)별 지표 표."""
    groups = defaultdict(list)
    for r in results:
        groups[(r.model_id, r.prompt_condition, r.test_name)].append(r)

    rows = []
    for (model, cond, test), rs in sorted(groups.items()):
        stat = unfaithfulness_rate(rs)
        lo, hi = bootstrap_ci(rs, n_boot=n_boot)
        row = {"model": model, "condition": cond, "test": test,
               **stat, "ci_low": lo, "ci_high": hi}
        if test == "masking":
            row.update(masking_control_stats(rs))
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# CoT vs post-hoc 비교 (RQ3 핵심)
# ---------------------------------------------------------------------------
def compare_conditions(results, test_name=None):
    """
    조건(cot vs posthoc)별 불충실률을 나란히 비교.
    반환: {model: {test: {cot: rate, posthoc: rate, diff: cot-posthoc}}}
    """
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        if test_name and r.test_name != test_name:
            continue
        groups[(r.model_id, r.test_name)][r.prompt_condition].append(r)

    out = {}
    for (model, test), conds in groups.items():
        row = {}
        for cond, rs in conds.items():
            row[cond] = unfaithfulness_rate(rs)["unfaithfulness_rate"]
        if "cot" in row and "posthoc" in row:
            row["diff_cot_minus_posthoc"] = row["cot"] - row["posthoc"]
        out.setdefault(model, {})[test] = row
    return out


def mcnemar_test(results_a, results_b):
    """
    McNemar 검정: 같은 케이스에 대한 두 조건의 불충실 판정이
    유의하게 다른가. scipy 없이 연속성 보정 McNemar를 직접 계산.
    """
    a_map, b_map = {}, {}
    for r in results_a:
        k = (r.case_id, r.detail.get("distractor") or r.detail.get("hint_label") or r.test_name)
        a_map[k] = r.is_unfaithful
    for r in results_b:
        k = (r.case_id, r.detail.get("distractor") or r.detail.get("hint_label") or r.test_name)
        b_map[k] = r.is_unfaithful

    keys = set(a_map) & set(b_map)
    b = sum(1 for k in keys if a_map[k] and not b_map[k])
    c = sum(1 for k in keys if not a_map[k] and b_map[k])
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "stat": float("nan"), "note": "불일치 쌍 없음"}
    stat = (abs(b - c) - 1) ** 2 / n
    sig = "유의(p<0.05)" if stat > 3.841 else "비유의"
    return {"b_a_only": b, "c_b_only": c, "n_discordant": n,
            "mcnemar_stat": stat, "note": sig}