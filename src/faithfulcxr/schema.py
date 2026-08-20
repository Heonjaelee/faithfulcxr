"""
FaithfulCXR 공통 데이터 스키마.

모든 모듈은 이 dataclass들을 주고받습니다. dict가 아니라 타입이 있는 객체를
쓰는 이유는, 필드 이름 오타나 누락을 조기에 잡고, 캐시 JSON의 구조를 강제하기
위해서입니다.

핵심 흐름:
    Case            하나의 흉부 X선 케이스 (이미지 경로 + 정답 라벨 + 병변 박스)
    Explanation     한 모델이 한 프롬프트 조건에서 낸 (예측 + 설명) 1건
    FaithResult     한 설명에 한 충실성 테스트를 적용한 결과 1건
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# 표준 진단 라벨 (closed set)
# ---------------------------------------------------------------------------
CANONICAL_LABELS = [
    "pneumonia",
    "pleural effusion",
    "cardiomegaly",
    "atelectasis",
    "consolidation",
    "pulmonary edema",
    "lung opacity",
    "pneumothorax",
    "no finding",
]


def canonicalize(text: str) -> str:
    """모델 자유텍스트 예측 -> 표준 라벨 정규화 (형식 미준수 시 안전장치)."""
    t = text.lower().strip()

    for lab in CANONICAL_LABELS:
        if t == lab:
            return lab

    keyword_map = [
        ("pneumonia",       ["pneumonia", "pneumonic"]),
        ("pneumothorax",    ["pneumothorax", "pneumothoraces"]),
        ("pleural effusion",["pleural effusion", "effusion"]),
        ("cardiomegaly",    ["cardiomegaly", "cardiomegalic", "enlarged heart",
                             "enlarged cardiac"]),
        ("pulmonary edema", ["pulmonary edema", "edema", "oedema"]),
        ("consolidation",   ["consolidation", "consolidative"]),
        ("atelectasis",     ["atelectasis", "atelectatic", "atelectases"]),
        ("lung opacity",    ["lung opacity", "airspace opacity", "opacities",
                             "opacity", "infiltrate", "infiltration"]),
    ]
    best_lab, best_pos = None, len(t) + 1
    for lab, keys in keyword_map:
        for k in keys:
            pos = t.find(k)
            if pos != -1 and pos < best_pos:
                best_pos, best_lab = pos, lab
    if best_lab:
        return best_lab

    if any(k in t for k in ["no acute", "no finding", "unremarkable",
                            "within normal", "no cardiopulmonary", "clear lung",
                            "normal"]):
        return "no finding"
    return "other"

# ---------------------------------------------------------------------------
# 1. Case: 데이터셋의 한 단위
# ---------------------------------------------------------------------------
@dataclass
class BBox:
    """병변 바운딩 박스 (픽셀 좌표). MS-CXR / VinDr에서 옴."""
    x: int
    y: int
    w: int
    h: int
    label: str            # 이 박스가 가리키는 소견 (예: "Pneumonia")


@dataclass
class Case:
    case_id: str
    image_path: str
    # 정답 라벨: 병리명 -> 존재여부/확실성. MIMIC-NLE는 3단계 확실성을 가짐.
    labels: dict[str, str] = field(default_factory=dict)   # {"Pneumonia": "positive"}
    # 리포트에서 파생된 "그럴듯한" 설명 (MIMIC-NLE). plausibility upper-bound 기준선.
    reference_nle: Optional[str] = None
    # 병변 박스들 (masking 테스트용). 없으면 빈 리스트.
    bboxes: list[BBox] = field(default_factory=list)
    # 계층 표본 관리를 위한 메타
    primary_pathology: Optional[str] = None
    certainty: Optional[str] = None      # high | medium | low

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Case":
        bboxes = [BBox(**b) for b in d.get("bboxes", [])]
        d = {**d, "bboxes": bboxes}
        return Case(**d)


# ---------------------------------------------------------------------------
# 2. Explanation: 한 모델이 한 조건에서 낸 출력
# ---------------------------------------------------------------------------
@dataclass
class Explanation:
    case_id: str
    model_id: str                 # chexagent2 | medgemma4b | gpt4o
    prompt_condition: str         # cot | posthoc
    prediction: str               # 모델이 내린 진단 라벨 (정규화된 문자열)
    explanation_text: str         # 자유 텍스트 설명
    raw_output: str = ""          # 파싱 전 원문 (디버깅용)

    def key(self) -> str:
        """캐시 키. 케이스 × 모델 × 조건으로 유일."""
        return f"{self.case_id}__{self.model_id}__{self.prompt_condition}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Explanation":
        return Explanation(**d)


# ---------------------------------------------------------------------------
# 3. FaithResult: 한 설명 + 한 테스트의 결과
# ---------------------------------------------------------------------------
@dataclass
class FaithResult:
    case_id: str
    model_id: str
    prompt_condition: str
    test_name: str                # counterfactual | biasing | masking
    # 개입/편향/마스킹 후 예측이 바뀌었는가?
    prediction_flipped: bool
    # 설명이 개입 요소(삽입 단어/힌트/영역)를 언급했는가?
    explanation_mentions: bool
    # 이 케이스가 "불충실"로 판정되는가?
    #   불충실 = 예측이 바뀌었는데(=근거가 바뀌었는데) 설명은 그걸 언급 안 함
    is_unfaithful: bool
    detail: dict = field(default_factory=dict)   # 테스트별 부가 정보

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FaithResult":
        return FaithResult(**d)


# ---------------------------------------------------------------------------
# JSONL 입출력 헬퍼 (캐시/결과 저장 공통)
# ---------------------------------------------------------------------------
def dump_jsonl(objs, path):
    """dataclass 리스트를 JSONL로 저장."""
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path, cls):
    """JSONL을 dataclass 리스트로 로드."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(cls.from_dict(json.loads(line)))
    return out
