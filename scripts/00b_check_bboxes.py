#!/usr/bin/env python
"""
스크립트 00b: 바운딩 박스 시각화 (좌표 정합성 확인).

마스킹 테스트의 전제: 박스가 실제 병변 위치에 정확히 놓여야 함.
박스를 이미지 위에 그려 results/bbox_check/ 에 저장합니다.

사용:
    python scripts/00b_check_bboxes.py --n 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image, ImageDraw

from faithfulcxr.utils.io import load_config
from faithfulcxr.data.loader import load_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["case_pool_size"] = args.n
    cases = load_cases(cfg)

    out_dir = os.path.join(cfg["paths"]["results"], "bbox_check")
    os.makedirs(out_dir, exist_ok=True)

    n_drawn = 0
    n_out_of_bounds = 0
    for case in cases[:args.n]:
        img = Image.open(case.image_path).convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img)
        for b in case.bboxes:
            if b.x < 0 or b.y < 0 or b.x + b.w > W or b.y + b.h > H:
                n_out_of_bounds += 1
            draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h],
                           outline=(255, 0, 0), width=3)
            draw.text((b.x + 2, max(0, b.y - 12)), b.label, fill=(255, 0, 0))
        out_path = os.path.join(out_dir, f"{case.case_id}.png")
        img.save(out_path)
        n_drawn += 1

    print(f"[bbox] {n_drawn}장 저장 -> {out_dir}")
    print(f"[bbox] 이미지 경계를 벗어난 박스: {n_out_of_bounds}개")
    if n_out_of_bounds > 0:
        print("[bbox] ⚠️ 박스가 이미지 밖으로 나갑니다 -> 좌표 스케일 문제!")
        print("       좌표까지 리사이즈된 데이터셋을 쓰거나 스케일 변환이 필요합니다.")
    else:
        print("[bbox] 경계 이탈 없음. 저장된 png를 열어 박스 위치가")
        print("       흉부의 그럴듯한 위치인지 눈으로 확인하세요.")
    print(f"\n확인: 파일 탐색기로 {out_dir} 폴더를 열어 몇 장 보세요.")


if __name__ == "__main__":
    main()