# scripts/00c_prepare_nih.py 로 저장
"""
NIH 원본에서 '박스 있는 이미지'만 추려서 data/raw/nih/ 에 정리.
로컬에서 한 번 실행 -> 프로젝트에 필요한 것만 남김 -> zip 크기 최소화.

사용:
    python scripts/00c_prepare_nih.py --src "다운로드받은_NIH_폴더" --out data/raw/nih
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="다운받은 NIH 원본 폴더")
    ap.add_argument("--out", default="data/raw/nih", help="정리 대상 폴더")
    args = ap.parse_args()

    # 1. BBox CSV 찾기
    bbox_csv = None
    for dp, _, files in os.walk(args.src):
        for f in files:
            if f.lower().replace("_","").replace(" ","") == "bboxlist2017.csv":
                bbox_csv = os.path.join(dp, f)
    if not bbox_csv:
        print("BBox_List_2017.csv를 못 찾음"); return
    print(f"CSV: {bbox_csv}")

    # 2. 박스에 등장하는 이미지 이름 수집
    needed = set()
    with open(bbox_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = (row.get("Image Index") or row.get("Image_Index")
                   or row.get("image"))
            if img:
                needed.add(img.strip())
    print(f"박스 있는 이미지: {len(needed)}개")

    # 3. 원본에서 그 이미지들만 복사
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    shutil.copy(bbox_csv, os.path.join(args.out, "BBox_List_2017.csv"))

    img_map = {}
    for dp, _, files in os.walk(args.src):
        for f in files:
            if f in needed:
                img_map[f] = os.path.join(dp, f)

    copied = 0
    for name in needed:
        if name in img_map:
            shutil.copy(img_map[name], os.path.join(args.out, "images", name))
            copied += 1
    print(f"복사 완료: {copied}/{len(needed)}개 -> {args.out}/images/")
    print(f"CSV -> {args.out}/BBox_List_2017.csv")


if __name__ == "__main__":
    main()