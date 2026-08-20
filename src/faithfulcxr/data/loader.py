"""
데이터 로더. config의 data.mode에 따라 세 소스 중 하나에서 Case 리스트를 만듭니다.

    mock   PhysioNet 승인 전 개발용. 합성 흉부 X선 이미지 + 가짜 라벨/박스/NLE.
           파이프라인 전체를 데이터 없이 검증할 수 있게 해줍니다.
    vindr  완전 공개 VinDr-CXR (방사선의 박스 보유). masking 테스트 프로토타이핑용.
    mimic  승인 후 실제 MIMIC-CXR + MIMIC-NLE + MS-CXR.

새 소스를 붙일 때는 _load_<mode>() 함수만 추가하면 됩니다.
"""
from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from faithfulcxr.schema import Case, BBox


# ---------------------------------------------------------------------------
# MOCK: 합성 데이터
# ---------------------------------------------------------------------------
def _make_synthetic_cxr(path: str, bboxes: list[BBox], seed: int, size: int = 512):
    """
    폐 실루엣처럼 보이는 회색조 이미지를 합성하고, 병변 박스 위치에
    밝은 반점을 그려 넣습니다. 진짜 X선은 아니지만, 이미지 파이프라인
    (로드/리사이즈/마스킹)을 검증하기엔 충분합니다.
    """
    rng = np.random.default_rng(seed)
    # 흉곽 배경: 가운데는 밝고 가장자리는 어두운 방사형 그라디언트
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2)
    base = np.clip(180 * (1 - r) + rng.normal(0, 12, (size, size)), 0, 255)
    img = Image.fromarray(base.astype(np.uint8), mode="L").convert("RGB")

    draw = ImageDraw.Draw(img)
    # 병변 위치에 밝은 반점 (소견을 흉내)
    for b in bboxes:
        for _ in range(60):
            px = b.x + rng.integers(0, max(1, b.w))
            py = b.y + rng.integers(0, max(1, b.h))
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(230, 230, 230))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    img.save(path)


def _load_mock(cfg) -> list[Case]:
    rng = random.Random(cfg["project"]["seed"])
    paths = cfg["paths"]
    img_dir = os.path.join(paths["data_processed"], "mock_images")
    os.makedirs(img_dir, exist_ok=True)

    pathologies = cfg["data"]["pathologies"]
    n = cfg["data"]["case_pool_size"]
    certainties = ["high", "medium", "low"]

    cases = []
    for i in range(n):
        cid = f"mock_{i:04d}"
        patho = pathologies[i % len(pathologies)]     # 병리 균등 분포
        cert = certainties[i % 3]
        # 병변 박스 하나를 무작위 위치에 (masking 테스트용)
        bx, by = rng.randint(80, 300), rng.randint(80, 300)
        bbox = BBox(x=bx, y=by, w=rng.randint(60, 120),
                    h=rng.randint(60, 120), label=patho)
        img_path = os.path.join(img_dir, f"{cid}.png")
        if not os.path.exists(img_path):
            _make_synthetic_cxr(img_path, [bbox], seed=i)
        # 리포트 파생 "그럴듯한" 설명을 흉내낸 문장
        ref_nle = (f"There is evidence of {patho.lower()} in the corresponding "
                   f"lung field, supporting the impression.")
        cases.append(Case(
            case_id=cid,
            image_path=img_path,
            labels={patho: "positive"},
            reference_nle=ref_nle,
            bboxes=[bbox],
            primary_pathology=patho,
            certainty=cert,
        ))
    return cases


# ---------------------------------------------------------------------------
# VINDR: 공개 데이터
# ---------------------------------------------------------------------------
def _load_vindr(cfg) -> list[Case]:
    """실제 VinDr-CXR 로더로 위임 (vindr_loader.py)."""
    from faithfulcxr.data.vindr_loader import _load_vindr as _real_vindr
    return _real_vindr(cfg)

def _load_nih_wrap(cfg) -> list[Case]:
    """NIH ChestX-ray14 로더로 위임 (nih_loader.py)."""
    from faithfulcxr.data.nih_loader import _load_nih
    return _load_nih(cfg)


# ---------------------------------------------------------------------------
# MIMIC: 승인 후 (스텁)
# ---------------------------------------------------------------------------
def _load_mimic(cfg) -> list[Case]:
    """
    MIMIC-CXR + MIMIC-NLE + MS-CXR 로더 스텁.
    실제 사용 시:
      1) MIMIC-NLE 추출 스크립트(공식 GitHub)를 MIMIC-CXR 리포트에 실행 ->
         (dicom_id, pathology, nle_text, certainty).
      2) MS-CXR의 phrase-bbox CSV에서 (dicom_id, bbox, label) 매칭.
      3) MIMIC-CXR-JPG의 이미지 경로 연결.
    지금은 데이터 접근 전이라 NotImplemented.
    """
    raise NotImplementedError(
        "MIMIC 로더는 PhysioNet 승인 + DUA 서명 후 활성화됩니다. "
        "그 전에는 config의 data.mode를 'mock' 또는 'vindr'로 두세요."
    )


LOADERS = {"mock": _load_mock, "vindr": _load_vindr,
           "nih": _load_nih_wrap, "mimic": _load_mimic}


def load_cases(cfg) -> list[Case]:
    mode = cfg["data"]["mode"]
    if mode not in LOADERS:
        raise ValueError(f"알 수 없는 data.mode: {mode}. {list(LOADERS)} 중 하나.")
    cases = LOADERS[mode](cfg)
    print(f"[loader] mode={mode}: {len(cases)}개 케이스 로드")
    return cases
