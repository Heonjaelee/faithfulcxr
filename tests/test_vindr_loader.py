"""
VinDr 로더 검증 테스트.

실제 데이터가 없어도, 미니 VinDr 형식 데이터(CSV + DICOM)를 임시로 만들어
파싱/변환/필터링/매핑이 올바른지 확인합니다.

    pytest tests/test_vindr_loader.py -v
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pydicom = pytest.importorskip("pydicom")  # pydicom 없으면 스킵
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from faithfulcxr.utils.io import load_config
from faithfulcxr.data.vindr_loader import load_vindr, VINDR_TO_CANON


def _make_dicom(path, seed):
    rng = np.random.default_rng(seed)
    arr = (rng.random((128, 128)) * 4000).astype(np.uint16)
    fm = Dataset()
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = 128, 128
    ds.PixelData = arr.tobytes()
    ds.save_as(path, write_like_original=False)


def test_vindr_loader_parses_and_converts():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = os.path.join(tmp, "train")
        proc_dir = os.path.join(tmp, "processed")
        os.makedirs(img_dir, exist_ok=True)

        ids = [f"img{i}" for i in range(4)]
        for i, iid in enumerate(ids):
            _make_dicom(os.path.join(img_dir, f"{iid}.dicom"), i)

        rows = []
        for i, iid in enumerate(ids):
            rows.append(dict(image_id=iid, rad_id="R1", class_name="Cardiomegaly",
                             x_min=10, y_min=10, x_max=60, y_max=80))
        # No finding 행 (필터링돼야 함)
        rows.append(dict(image_id="img0", rad_id="R2", class_name="No finding",
                         x_min=np.nan, y_min=np.nan, x_max=np.nan, y_max=np.nan))
        ann_path = os.path.join(tmp, "ann.csv")
        pd.DataFrame(rows).to_csv(ann_path, index=False)

        cfg = load_config("configs/config.yaml")
        cfg["data"]["mode"] = "vindr"
        cfg["data"]["case_pool_size"] = 10
        cfg["data"]["vindr_ann"] = ann_path
        cfg["data"]["vindr_img"] = img_dir
        cfg["paths"]["data_processed"] = proc_dir

        cases = load_vindr(cfg)

        # 4개 이미지 모두 케이스로
        assert len(cases) == 4
        # DICOM -> PNG 변환됨
        assert all(c.image_path.endswith(".png") for c in cases)
        assert all(os.path.exists(c.image_path) for c in cases)
        # 박스가 (x,y,w,h)로 변환됨
        b = cases[0].bboxes[0]
        assert b.w == 50 and b.h == 70          # 60-10, 80-10
        # 클래스 매핑
        assert b.label == VINDR_TO_CANON["Cardiomegaly"]
        # No finding 행은 박스로 안 들어감 (img0도 박스 1개만)
        img0 = [c for c in cases if c.case_id == "vindr_img0"][0]
        assert len(img0.bboxes) == 1