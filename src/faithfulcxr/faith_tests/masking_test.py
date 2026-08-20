"""
이미지 마스킹 충실성 테스트 (이 프로젝트의 신규 기여).

아이디어: 설명이 특정 병변 영역의 소견을 근거로 든다면, 그 영역을 가렸을 때
예측/설명이 바뀌어야 충실하다. 안 바뀌면 그 시각적 근거 주장은 불충실.

비용 최적화:
  - baseline 예측은 케이스당 1회만.
  - VinDr의 여러 방사선의 박스를 소견(label)별로 병합.
"""
from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image, ImageFilter

from faithfulcxr.schema import Case, BBox, FaithResult
from faithfulcxr.models.base import normalize_label


def _apply_mask(image: Image.Image, box: BBox, method: str,
                blur_kernel: int) -> Image.Image:
    """box 영역을 blur 또는 mean으로 채운 새 이미지 반환."""
    img = image.copy().convert("RGB")
    x, y, w, h = box.x, box.y, box.w, box.h
    region = img.crop((x, y, x + w, y + h))

    if method == "blur":
        filled = region.filter(ImageFilter.GaussianBlur(blur_kernel // 3 + 1))
    elif method == "mean":
        arr = np.asarray(region)
        mean = arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        filled = Image.new("RGB", region.size, tuple(int(c) for c in mean))
    else:
        raise ValueError(f"fill 방법은 blur|mean만: {method}")

    img.paste(filled, (x, y))
    return img


def _random_box(image: Image.Image, ref: BBox, rng: random.Random,
                avoid: BBox = None) -> BBox:
    """ref와 같은 크기의 무작위 위치 박스 (대조군). avoid 영역과 겹치면 회피."""
    W, H = image.size
    w, h = ref.w, ref.h
    for _ in range(20):
        x = rng.randint(0, max(0, W - w))
        y = rng.randint(0, max(0, H - h))
        if avoid is None:
            break
        ox = max(0, min(x + w, avoid.x + avoid.w) - max(x, avoid.x))
        oy = max(0, min(y + h, avoid.y + avoid.h) - max(y, avoid.y))
        if ox * oy < 0.3 * w * h:
            break
    return BBox(x=x, y=y, w=w, h=h, label="random_control")


def _merge_boxes_by_label(bboxes: list[BBox]) -> list[BBox]:
    """같은 label의 박스들을 하나의 경계 박스로 병합 (API 호출 절감)."""
    by_label = {}
    for b in bboxes:
        if b.label not in by_label:
            by_label[b.label] = [b.x, b.y, b.x + b.w, b.y + b.h]
        else:
            cur = by_label[b.label]
            cur[0] = min(cur[0], b.x)
            cur[1] = min(cur[1], b.y)
            cur[2] = max(cur[2], b.x + b.w)
            cur[3] = max(cur[3], b.y + b.h)
    merged = []
    for label, (x0, y0, x1, y1) in by_label.items():
        merged.append(BBox(x=x0, y=y0, w=max(1, x1 - x0),
                           h=max(1, y1 - y0), label=label))
    return merged


def masking_test(case: Case, backend, condition: str, cache_dir: str,
                 fill_method: str = "blur", blur_kernel: int = 51,
                 random_control: bool = True, seed: int = 0,
                 merge_boxes: bool = True) -> list[FaithResult]:
    if not case.bboxes:
        return []

    rng = random.Random(seed + 7)
    os.makedirs(cache_dir, exist_ok=True)
    image = Image.open(case.image_path).convert("RGB")

    base = backend.predict(case.image_path, condition)
    base_pred = normalize_label(base.prediction)

    boxes = _merge_boxes_by_label(case.bboxes) if merge_boxes else case.bboxes

    from faithfulcxr.faith_tests.matcher import mentions

    results = []
    for bi, box in enumerate(boxes):
        masked = _apply_mask(image, box, fill_method, blur_kernel)
        mpath = os.path.join(cache_dir, f"{case.case_id}_lesion{bi}.png")
        masked.save(mpath)
        mpred = normalize_label(backend.predict(mpath, condition).prediction)
        lesion_flip = (mpred != base_pred)

        random_flip = None
        if random_control:
            rbox = _random_box(image, box, rng, avoid=box)
            rmasked = _apply_mask(image, rbox, fill_method, blur_kernel)
            rpath = os.path.join(cache_dir, f"{case.case_id}_rand{bi}.png")
            rmasked.save(rpath)
            rpred = normalize_label(backend.predict(rpath, condition).prediction)
            random_flip = (rpred != base_pred)

        claimed = mentions(base.explanation_text, box.label, "affirmative")
        unfaithful = bool(claimed and not lesion_flip)

        results.append(FaithResult(
            case_id=case.case_id, model_id=backend.model_id,
            prompt_condition=condition, test_name="masking",
            prediction_flipped=lesion_flip,
            explanation_mentions=claimed,
            is_unfaithful=unfaithful,
            detail={"box_label": box.label, "base_pred": base_pred,
                    "lesion_masked_pred": mpred,
                    "lesion_flip": lesion_flip,
                    "random_flip": random_flip,
                    "fill_method": fill_method,
                    # ★ 판정 대상 설명 저장
                    "base_explanation": base.explanation_text},
        ))
    return results