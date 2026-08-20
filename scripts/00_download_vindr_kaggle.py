#!/usr/bin/env python
"""
스크립트 00: Kaggle에서 VinDr-CXR 데이터 일부만 받기.

원본 대회 데이터는 DICOM ~200GB라 통째로만 받아짐 -> 부담 큼.
대신 커뮤니티 경량 PNG 재배포본을 받고, 그중 N장만 샘플링해서
data/raw/vindr/ 에 우리 로더 형식으로 배치합니다.

사전 준비 (한 번만):
    1) pip install kaggle
    2) Kaggle 계정 -> Settings -> API -> "Create New Token" -> kaggle.json
    3) Windows: C:\\Users\\<사용자>\\.kaggle\\kaggle.json 에 배치
    4) 대회 규칙에 동의:
       https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/rules

사용:
    python scripts/00_download_vindr_kaggle.py --n 300
    python scripts/00_download_vindr_kaggle.py --n 300 --keep-full
"""
import argparse
import os
import shutil
import subprocess
import sys

RAW_DIR = "data/raw/vindr"
KAGGLE_DATASET = "awsaf49/vinbigdata-512-image-dataset"


def run(cmd):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="남길 이미지 수")
    ap.add_argument("--keep-full", action="store_true", help="원본 zip 유지")
    ap.add_argument("--dataset", default=KAGGLE_DATASET)
    args = ap.parse_args()

    try:
        import kaggle  # noqa
    except ImportError:
        print("kaggle 미설치. 먼저: pip install kaggle")
        sys.exit(1)

    os.makedirs(RAW_DIR, exist_ok=True)
    dl_dir = os.path.join(RAW_DIR, "_download")
    os.makedirs(dl_dir, exist_ok=True)

    print(f"[dl] {args.dataset} 다운로드 중... (수 GB, 최초 1회)")
    run(["kaggle", "datasets", "download", "-d", args.dataset,
         "-p", dl_dir, "--unzip"])

    csv_path, img_dir = _find_csv_and_images(dl_dir)
    if csv_path is None:
        print("[!] train.csv를 찾지 못했습니다. 다운로드 폴더를 확인하세요:")
        print(f"    {dl_dir}")
        sys.exit(1)
    print(f"[dl] csv={csv_path}")
    print(f"[dl] images={img_dir}")

    _prepare(csv_path, img_dir, n=args.n)

    if not args.keep_full:
        print("[clean] 원본 다운로드 폴더 삭제(용량 절약)...")
        shutil.rmtree(dl_dir, ignore_errors=True)

    print("\n완료. 이제 config.yaml에서 data.mode: vindr 로 바꾸고:")
    print("  python scripts/01_generate_explanations.py --mock --limit 20")


def _find_csv_and_images(root: str):
    """다운로드 폴더에서 train.csv와 png가 가장 많은 폴더를 찾음."""
    csv_path, img_dir, best_count = None, None, 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == "train.csv":
                csv_path = os.path.join(dirpath, f)
        n_img = sum(1 for f in files if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if n_img > best_count:
            best_count = n_img
            img_dir = dirpath
    return csv_path, img_dir


def _prepare(csv_path: str, img_dir: str, n: int):
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "class_name" in df.columns:
        df = df[df["class_name"].str.lower() != "no finding"]
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])

    def img_exists(iid):
        for ext in (".png", ".jpg", ".jpeg"):
            if os.path.exists(os.path.join(img_dir, iid + ext)):
                return True
        return False

    ids = [i for i in df["image_id"].unique() if img_exists(i)][:n]
    df = df[df["image_id"].isin(ids)]
    print(f"[prep] {len(ids)}개 이미지 / {len(df)}개 박스 선택")

    out_img = os.path.join(RAW_DIR, "train")
    os.makedirs(out_img, exist_ok=True)
    copied = 0
    for iid in ids:
        for ext in (".png", ".jpg", ".jpeg"):
            src = os.path.join(img_dir, iid + ext)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_img, iid + ext))
                copied += 1
                break
    print(f"[prep] {copied}개 이미지 복사 -> {out_img}")

    out_csv = os.path.join(RAW_DIR, "annotations_train.csv")
    # 원본 크기 컬럼(width/height 등)을 반드시 보존 -> 좌표 스케일에 필요
    size_cols = ["width", "height", "original_width", "original_height",
                 "dim0", "dim1", "img_width", "img_height", "rows", "columns"]
    keep_cols = [c for c in (["image_id", "rad_id", "class_name",
                              "x_min", "y_min", "x_max", "y_max"] + size_cols)
                 if c in df.columns]
    df[keep_cols].to_csv(out_csv, index=False)
    if not any(c in df.columns for c in size_cols):
        print("[prep] ⚠️ 원본 CSV에 크기 컬럼이 없습니다. 좌표 스케일이 불가할 수 있어요.")
    else:
        found = [c for c in size_cols if c in df.columns]
        print(f"[prep] 크기 컬럼 보존됨 {found} -> 좌표 스케일 가능")
    print(f"[prep] annotations -> {out_csv}")


if __name__ == "__main__":
    main()