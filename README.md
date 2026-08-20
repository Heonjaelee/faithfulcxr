# FaithfulCXR

**흉부 X선 자연어 설명의 충실성(Faithfulness) 벤치마크**

방사선 VLM이 내놓는 설명이 단지 *그럴듯한(plausible)* 것인지, 실제 판단 근거를
반영하는 *충실한(faithful)* 것인지를 측정합니다. 학습 없이 추론·API 기반으로
동작하여 노트북(RTX 3050)에서 실행 가능합니다.

---

## 핵심 아이디어

- **3모델 (도메인 특화 축):** CheXagent-2(CXR 전용) / MedGemma-4B(범용 의료) / GPT-4o(비의료 범용)
- **2프롬프트:** CoT(설명→진단) / post-hoc(진단→설명)
- **3충실성 테스트:** counterfactual / biasing / image-masking(MS-CXR 병변 박스)
- **그럴듯함 축:** MIMIC-NLE 리포트 기반 기준선 + LLM-as-judge (의사 섭외 불필요)

---

## 설치

```bash
# 가상환경 권장
python -m venv .venv && source .venv/bin/activate

# 핵심만 (분석/파이프라인 검증용, mock 모드로 바로 돌릴 수 있음)
pip install pyyaml numpy pillow

# 전체 (로컬 HF 4-bit + API). torch는 본인 CUDA에 맞게 pytorch.org에서 설치
pip install -r requirements.txt
```

RTX 3050 사용자는 `torch`를 반드시 **CUDA 빌드**로 설치하세요
(예: `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

---

## 실행 순서 (지금 바로 — 데이터 없이 mock으로)

PhysioNet 승인 전에도 파이프라인 전체를 검증할 수 있습니다.

```bash
# 0) 스모크 테스트 — 전부 초록불인지 확인
python -m pytest tests/ -v

# 1) 설명 생성 (mock 모델, 파일럿 20케이스)
python scripts/01_generate_explanations.py --mock --limit 20

# 2) 충실성 테스트 3종 실행
python scripts/02_run_faith_tests.py --mock --limit 20

# 3) 분석 — 불충실 비율 표 + results/summary.csv
python scripts/03_analyze.py
```

mock 모드에서는 세 모델이 동일 결과를 냅니다(같은 가짜 백엔드).
이는 정상이며, **파이프라인 구조를 검증**하는 것이 목적입니다.

---

## 실제 모델·데이터로 전환

### (A) 실제 모델 켜기
`configs/config.yaml`의 각 모델 `backend`는 이미 설정돼 있습니다.
`--mock` 플래그만 빼면 실제 백엔드를 씁니다.

- 로컬 HF(MedGemma/CheXagent): `hf_backend.py`의 모델 로드 두 줄을
  각 모델 카드의 예제 코드로 교체(모델마다 processor/model 클래스가 다름).
- API(GPT-4o): 환경변수 `OPENAI_API_KEY` 설정.
  ⚠️ MIMIC 데이터를 API로 보낼 때는 **zero-data-retention** 확인 필수.

### (B) 실제 데이터 켜기
`configs/config.yaml`의 `data.mode`를 바꿉니다.

- `mock` → 합성 데이터 (기본, 승인 전)
- `vindr` → 공개 VinDr-CXR (`loader.py::_load_vindr` 채우기)
- `mimic` → 승인 후 MIMIC-CXR+NLE+MS-CXR (`loader.py::_load_mimic` 채우기)

MIMIC 로더는 (1) MIMIC-NLE 추출 스크립트 실행, (2) MS-CXR 박스 매칭,
(3) MIMIC-CXR-JPG 이미지 경로 연결로 구성됩니다.

---

## 8주 계획과 코드 매핑

| 주차 | 할 일 | 사용 파일 |
|---|---|---|
| 1 | 환경 세팅, mock 파이프라인 검증, 모델 스모크 | `tests/`, `scripts/01 --mock` |
| 2 | 실제 데이터 로더 채우기, 케이스 풀 확정 | `data/loader.py` |
| 3 | 설명 생성·캐싱 (6조건) | `scripts/01` |
| 4 | counterfactual 테스트 | `faith_tests/text_tests.py`, `scripts/02 --tests counterfactual` |
| 5 | biasing + masking 테스트 | `faith_tests/`, `scripts/02` |
| 6 | LLM-judge 그럴듯함 (추가 예정: `scripts/04`) | `faith_tests/matcher.py` |
| 7 | 분석·시각화·통계 | `analysis/metrics.py`, `scripts/03` |
| 8 | 작성·코드 정리 | — |

**우선순위 버퍼:** 텍스트 기반 2종(counterfactual+biasing)이 보호되는 핵심.
masking은 신규성 최고이나 리스크도 최고 → 추가 기여로 취급.

---

## 폴더 구조

```
faithfulcxr/
├── configs/config.yaml          # 모든 실험 파라미터
├── scripts/
│   ├── 01_generate_explanations.py
│   ├── 02_run_faith_tests.py
│   └── 03_analyze.py
├── src/faithfulcxr/
│   ├── schema.py                # Case / Explanation / FaithResult
│   ├── data/loader.py           # mock | vindr | mimic
│   ├── models/
│   │   ├── base.py              # 추상 백엔드 + mock + 프롬프트/파싱
│   │   ├── hf_backend.py        # 로컬 4-bit
│   │   └── api_backend.py       # GPT-4o 등
│   ├── faith_tests/
│   │   ├── matcher.py           # "설명이 X를 언급했나" (문자열/의미)
│   │   ├── text_tests.py        # counterfactual + biasing
│   │   └── masking_test.py      # 이미지 마스킹 (+무작위 대조군)
│   └── analysis/metrics.py      # 불충실 비율 + bootstrap CI
├── tests/test_pipeline.py       # 스모크 테스트
├── data/  results/              # 캐시·산출물 (git 제외)
└── requirements.txt
```

---

## 다음 구현 예정 (스캐폴딩만 있고 채워야 할 것)

1. `data/loader.py::_load_vindr` / `_load_mimic` — 실제 데이터 파싱
2. `scripts/04_judge_plausibility.py` — LLM-as-judge 그럴듯함 점수 (matcher의 judge_fn 재사용)
3. `analysis/` — 그럴듯함×충실성 사분면 플롯, McNemar 검정, Adebayo sanity check
4. `hf_backend.py` — 각 모델 카드에 맞춘 정확한 로드/디코드 코드
