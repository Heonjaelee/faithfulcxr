"""
NIH ChestX-ray14 로더 (공개 데이터, 크리덴셜 불필요).
tensorflow.csv 형식: filename, width, height, class, xmin, ymin, xmax, ymax
⚠️ CheXagent 학습 데이터. VinDr와 비교용.
"""
from __future__ import annotations

import os
import csv

from faithfulcxr.schema import Case, BBox

NIH_TO_CANON = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Infiltrate": "Consolidation",
    "Infiltration": "Consolidation",
    "Mass": "Lung Opacity",
    "Nodule": "Lung Opacity",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}

_CSV_CANDIDATES = ["tensorflow.csv", "BBox_List_2017.csv"]


def _find_csv(root: str) -> str:
    for name in _CSV_CANDIDATES:
        for dp, _, files in os.walk(root):
            for f in files:
                if f.lower() == name.lower():
                    return os.path.join(dp, f)
    raise FileNotFoundError(
        f"NIH CSV(tensorflow.csv 또는 BBox_List_2017.csv)를 {root}에서 못 찾음.")


def _build_image_index(root: str) -> dict:
    idx = {}
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                idx[f] = os.path.join(dp, f)
    return idx


def _load_nih(cfg) -> list[Case]:
    paths = cfg["paths"]
    root = cfg["data"].get("nih_root") or os.path.join(paths["data_raw"], "nih")
    size = cfg["data"].get("image_size", 512)
    limit = cfg["data"].get("case_pool_size", 300)

    csv_path = _find_csv(root)
    img_index = _build_image_index(root)
    print(f"[nih] CSV: {csv_path}")
    print(f"[nih] 이미지 인덱스: {len(img_index)}개 파일")

    is_tf = os.path.basename(csv_path).lower() == "tensorflow.csv"

    per_image = {}
    n_scaled = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_tf:
                img = (row.get("filename") or "").strip()
                label = (row.get("class") or "").strip()
                try:
                    ow = float(row["width"]); oh = float(row["height"])
                    xmin = float(row["xmin"]); ymin = float(row["ymin"])
                    xmax = float(row["xmax"]); ymax = float(row["ymax"])
                except (KeyError, ValueError, TypeError):
                    continue
                x, y = xmin, ymin
                w, h = xmax - xmin, ymax - ymin
            else:
                img = (row.get("Image Index") or "").strip()
                label = (row.get("Finding Label") or "").strip()
                keys = list(row.keys())
                try:
                    vals = [row[k] for k in keys[2:6]]
                    x, y, w, h = [float(v) for v in vals]
                    ow = oh = 1024.0
                except (ValueError, TypeError, IndexError):
                    continue

            if not img or label not in NIH_TO_CANON:
                continue
            canon = NIH_TO_CANON[label]
            sx = size / ow
            sy = size / oh
            bb = BBox(x=int(x*sx), y=int(y*sy),
                      w=int(w*sx), h=int(h*sy), label=canon)
            per_image.setdefault(img, []).append(bb)
            n_scaled += 1

    print(f"[nih] 원본→512 스케일 적용: {n_scaled}개 박스")
    print(f"[nih] 박스 있는 이미지: {len(per_image)}개")

    cases = []
    for img_name, bboxes in per_image.items():
        if img_name not in img_index:
            continue
        primary = bboxes[0].label
        cases.append(Case(
            case_id=f"nih_{os.path.splitext(img_name)[0]}",
            image_path=img_index[img_name],
            labels={b.label: "positive" for b in bboxes},
            reference_nle=None,
            bboxes=bboxes,
            primary_pathology=primary,
            certainty="high",
        ))
        if len(cases) >= limit:
            break

    print(f"[nih] {len(cases)}개 케이스 로드 (박스 기반)")
    return cases