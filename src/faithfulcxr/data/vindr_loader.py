"""
VinDr-CXR 로더 (공개 데이터).

데이터 출처 (둘 중 하나):
  - Kaggle "VinBigData Chest X-ray Abnormalities Detection" (권장, 접근 쉬움)
        train.csv 컬럼: image_id, class_name, class_id, rad_id,
                        x_min, y_min, x_max, y_max
        이미지: train/<image_id>.dicom  (일부 재배포판은 PNG/JPG)
  - PhysioNet vindr-cxr v1.0.0 (DUA 필요)
        annotations_train.csv 컬럼: image_id, rad_id, class_name,
                                    x_min, y_min, x_max, y_max
        이미지: train/<image_id>.dicom

두 형식 모두 컬럼 이름이 사실상 같아, 아래 파서 하나로 처리합니다.

이미지 처리:
  - .dicom -> pydicom으로 읽어 png로 변환 후 캐시 (VLM 백엔드는 png를 먹음)
  - 이미 png/jpg면 그대로 사용

바운딩 박스:
  - (x_min,y_min,x_max,y_max) -> 스키마의 BBox(x,y,w,h)로 변환
  - 'No finding' 행은 박스가 없으므로 제외
  - 여러 방사선의가 같은 소견에 박스를 그리므로, 케이스당 첫 박스만 쓰거나
    (기본) 소견별로 대표 박스 하나를 선택
"""
from __future__ import annotations

import os

from faithfulcxr.schema import Case, BBox


# VinDr 22개 local finding 중, 우리 8종 심폐 소견에 매핑되는 것 위주로 필터.
# (MS-CXR 8종과 어휘를 맞춰 나중에 MIMIC 전환 시 일관성 유지)
VINDR_TO_CANON = {
    "Aortic enlargement": "Cardiomegaly",       # 근사 매핑 (심장·대혈관 계열)
    "Cardiomegaly": "Cardiomegaly",
    "Pleural effusion": "Pleural Effusion",
    "Pleural thickening": "Pleural Effusion",
    "Pneumothorax": "Pneumothorax",
    "Infiltration": "Consolidation",
    "Consolidation": "Consolidation",
    "Lung Opacity": "Lung Opacity",
    "Atelectasis": "Atelectasis",
    "Pulmonary fibrosis": "Lung Opacity",
    "ILD": "Lung Opacity",
    "Nodule/Mass": "Lung Opacity",
    # 그 외 소견은 무시 (No finding, Calcification 등)
}


def _dicom_to_png(dicom_path: str, out_path: str):
    """DICOM -> png 변환 (window 적용, 8bit 정규화)."""
    import numpy as np
    import pydicom
    from PIL import Image

    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(float)

    # MONOCHROME1이면 반전 (밝고 어두움 관례가 반대)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    # 0~255 정규화
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)

    Image.fromarray(arr).convert("RGB").save(out_path)


def _resolve_image(image_id: str, img_dir: str, png_cache: str) -> str | None:
    """
    image_id에 해당하는 이미지 파일을 찾고, DICOM이면 png로 변환해 경로 반환.
    없으면 None.
    """
    os.makedirs(png_cache, exist_ok=True)
    png_path = os.path.join(png_cache, f"{image_id}.png")
    if os.path.exists(png_path):
        return png_path

    # 이미 png/jpg로 배포된 경우
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(img_dir, image_id + ext)
        if os.path.exists(p):
            return p

    # DICOM인 경우 변환
    for ext in (".dicom", ".dcm"):
        p = os.path.join(img_dir, image_id + ext)
        if os.path.exists(p):
            _dicom_to_png(p, png_path)
            return png_path

    return None

def _find_orig_size(row, df_cols):
    """
    CSV 행에서 원본 이미지 크기를 찾음. 여러 관례적 컬럼명 대응.
    (width,height) | (dim1,dim0) | (original_width,original_height) 등.
    없으면 (None, None).
    """
    candidates = [
        ("width", "height"),
        ("original_width", "original_height"),
        ("dim1", "dim0"),          # dim0=height, dim1=width 관례
        ("img_width", "img_height"),
        ("w_orig", "h_orig"),
    ]
    for wc, hc in candidates:
        if wc in df_cols and hc in df_cols:
            try:
                w = float(row[wc]); h = float(row[hc])
                if w > 0 and h > 0:
                    return w, h
            except (ValueError, TypeError):
                pass
    return None, None

def load_vindr(cfg) -> list[Case]:
    """
    config에서 아래 경로를 읽습니다:
        data.vindr_ann   annotations CSV 경로
        data.vindr_img   이미지 폴더 (dicom 또는 png)

    좌표 스케일:
        재배포 512/1024 png는 리사이즈됐는데 CSV 좌표가 원본 기준일 수 있음.
        CSV에 원본 크기 컬럼(width/height 등)이 있으면 그 비율로 변환.
    """
    import pandas as pd
    from PIL import Image

    dcfg = cfg["data"]
    ann_path = dcfg.get("vindr_ann", "data/raw/vindr/annotations_train.csv")
    img_dir = dcfg.get("vindr_img", "data/raw/vindr/train")
    png_cache = os.path.join(cfg["paths"]["data_processed"], "vindr_png")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"VinDr annotation CSV를 찾을 수 없습니다: {ann_path}\n"
            "config.yaml의 data.vindr_ann / data.vindr_img 경로를 설정하거나,\n"
            "data/raw/vindr/ 아래에 annotations_train.csv와 train/ 폴더를 두세요."
        )

    df = pd.read_csv(ann_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df_cols = set(df.columns)
    required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    missing = required - df_cols
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])
    df = df[df["class_name"].isin(VINDR_TO_CANON.keys())]

    n_target = cfg["data"]["case_pool_size"]
    cases = []
    n_scaled = 0
    for image_id, grp in df.groupby("image_id"):
        if len(cases) >= n_target:
            break
        img_path = _resolve_image(image_id, img_dir, png_cache)
        if img_path is None:
            continue

        with Image.open(img_path) as im:
            png_w, png_h = im.size

        bboxes = []
        labels = {}
        for _, row in grp.iterrows():
            canon = VINDR_TO_CANON[row["class_name"]]
            x0, y0, x1, y1 = (float(row["x_min"]), float(row["y_min"]),
                              float(row["x_max"]), float(row["y_max"]))

            ow, oh = _find_orig_size(row, df_cols)
            if ow and oh:
                sx, sy = png_w / ow, png_h / oh
                x0, x1 = x0 * sx, x1 * sx
                y0, y1 = y0 * sy, y1 * sy
                n_scaled += 1

            x0 = max(0, min(x0, png_w - 1)); x1 = max(0, min(x1, png_w))
            y0 = max(0, min(y0, png_h - 1)); y1 = max(0, min(y1, png_h))
            xi, yi = int(x0), int(y0)
            wi, hi = max(1, int(x1 - x0)), max(1, int(y1 - y0))

            bboxes.append(BBox(x=xi, y=yi, w=wi, h=hi, label=canon))
            labels[canon] = "positive"

        if not bboxes:
            continue
        primary = bboxes[0].label
        cases.append(Case(
            case_id=f"vindr_{image_id}",
            image_path=img_path,
            labels=labels,
            reference_nle=None,
            bboxes=bboxes,
            primary_pathology=primary,
            certainty="high",
        ))

    if not cases:
        raise RuntimeError(
            "VinDr 케이스를 하나도 만들지 못했습니다. 이미지 경로/형식을 확인하세요."
        )
    if n_scaled > 0:
        print(f"[vindr] 원본 크기 컬럼으로 좌표 스케일 적용: {n_scaled}개 박스")
    else:
        print("[vindr] ⚠️ 원본 크기 컬럼이 없어 스케일 미적용, 클리핑만 함. "
              "박스가 어긋나면 크기 컬럼 있는 CSV가 필요합니다.")
    return cases


# loader.py가 기대하는 이름과의 alias
def _load_vindr(cfg):
    return load_vindr(cfg)