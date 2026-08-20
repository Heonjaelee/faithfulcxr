"""공통 유틸: config 로드, 시드 고정, 캐시 경로."""
from __future__ import annotations

import os
import random

import yaml


def load_dotenv(path: str = ".env"):
    """
    프로젝트 루트의 .env 에서 KEY=VALUE 를 읽어 환경변수로 설정.
    라이브러리 없이 처리. 이미 설정된 값은 덮어쓰지 않음.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def load_config(path: str = "configs/config.yaml") -> dict:
    load_dotenv()   # config 로드 시 .env 자동 로드
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def ensure_dirs(cfg: dict):
    for key in ("data_raw", "data_processed", "cache", "results"):
        os.makedirs(cfg["paths"][key], exist_ok=True)
