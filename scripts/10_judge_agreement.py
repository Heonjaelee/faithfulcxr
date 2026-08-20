#!/usr/bin/env python
"""
스크립트 10: judge 간 일치도 분석 + 평균 그럴듯함 산출.

여러 judge로 매긴 그럴듯함 파일들을 받아:
  1) judge 간 일치도(Spearman ρ, Pearson r, 평균 절대차) 계산
  2) self-preference 편향 점검: 각 judge가 각 대상 모델을 어떻게 봤는지
  3) 메인 그럴듯함 = 지정한 judge들의 평균 -> 병합 파일로 저장

사용:
  python scripts/10_judge_agreement.py \
      --judges gemini=results/plausibility_gpt_gemini.jsonl \
               claude=results/plausibility_gpt_claude.jsonl \
      --compare gpt=results/plausibility_gpt.jsonl \
      --output results/plausibility.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _load(path):
    """key -> record. key=(case,model,condition)."""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[(r["case_id"], r["model_id"], r["prompt_condition"])] = r
    return d


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs) ** 0.5
    vy = sum((y-my)**2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)


def _spearman(xs, ys):
    """scipy 없이 Spearman ρ (순위 상관)."""
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j+1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    return _pearson(rx, ry)


def _parse_pairs(items):
    out = {}
    for it in items:
        name, path = it.split("=", 1)
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True,
                    help="메인 judge들: name=path (예: gemini=... claude=...)")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="비교군 judge(편향 확인용, 평균 미포함): name=path")
    ap.add_argument("--output", default=None,
                    help="평균 그럴듯함 저장 경로 (메인 judge 평균)")
    args = ap.parse_args()

    judges = {name: _load(path) for name, path in _parse_pairs(args.judges).items()}
    compares = {name: _load(path) for name, path in _parse_pairs(args.compare).items()}

    names = list(judges)
    print(f"[agree] 메인 judge: {names}")
    if compares:
        print(f"[agree] 비교군: {list(compares)}")

    common = set.intersection(*[set(d) for d in judges.values()])
    print(f"[agree] 공통 평가 항목: {len(common)}건\n")

    # 1) judge 쌍별 일치도
    print("=== judge 간 일치도 ===")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            xs = [judges[a][k]["plausibility"] for k in common]
            ys = [judges[b][k]["plausibility"] for k in common]
            rho = _spearman(xs, ys)
            r = _pearson(xs, ys)
            mad = sum(abs(x-y) for x, y in zip(xs, ys)) / len(xs)
            print(f"  {a} vs {b}: Spearman ρ={rho:.3f}  Pearson r={r:.3f}  "
                  f"평균절대차={mad:.2f}")

    # 2) 각 judge가 본 모델별 평균 (self-preference 점검)
    print("\n=== judge별 × 대상모델별 평균 그럴듯함 ===")
    all_judges = {**judges, **compares}
    models = sorted(set(k[1] for k in common))
    header = "judge".ljust(12) + "".join(m.ljust(14) for m in models)
    print(header)
    for jname, jd in all_judges.items():
        row = jname.ljust(12)
        for m in models:
            vals = [jd[k]["plausibility"] for k in jd
                    if k[1] == m and jd[k]["plausibility"] == jd[k]["plausibility"]]
            avg = sum(vals)/len(vals) if vals else float("nan")
            row += f"{avg:.2f}".ljust(14)
        print(row)
    if compares:
        print("  (비교군 judge가 특정 모델을 유독 높게 주면 self-preference 의심)")

    # 3) 메인 judge 평균 -> 병합 저장
    if args.output:
        merged = []
        for k in common:
            scores = [judges[n][k]["plausibility"] for n in names
                      if judges[n][k]["plausibility"] == judges[n][k]["plausibility"]]
            if not scores:
                continue
            rec = dict(judges[names[0]][k])
            rec["plausibility"] = sum(scores) / len(scores)
            rec["plausibility_by_judge"] = {n: judges[n][k]["plausibility"]
                                            for n in names}
            merged.append(rec)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        avg_all = sum(r["plausibility"] for r in merged) / len(merged)
        print(f"\n[agree] 메인 judge 평균 저장 -> {args.output}")
        print(f"[agree] {len(merged)}건, 전체 평균 그럴듯함 {avg_all:.2f}")


if __name__ == "__main__":
    main()