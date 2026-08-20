"""
진행 상황 표시 유틸.

tqdm이 설치돼 있으면 예쁜 진행바를, 없으면 표준 라이브러리만으로
간단한 진행률/ETA를 출력합니다. 코드는 tqdm 유무를 신경 안 써도 됩니다.

사용:
    from faithfulcxr.utils.progress import progress
    for item in progress(items, desc="설명 생성"):
        ...
"""
from __future__ import annotations

import sys
import time


def _fallback_progress(iterable, total=None, desc="진행"):
    """tqdm 없을 때: 표준 출력에 간단한 진행률 + 경과/ETA를 갱신."""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    start = time.time()
    count = 0
    last_len = 0
    for item in iterable:
        yield item
        count += 1
        elapsed = time.time() - start
        rate = count / elapsed if elapsed > 0 else 0
        if total:
            frac = count / total
            bar_w = 24
            filled = int(bar_w * frac)
            bar = "#" * filled + "-" * (bar_w - filled)
            eta = (total - count) / rate if rate > 0 else 0
            msg = (f"\r{desc}: [{bar}] {count}/{total} "
                   f"({frac*100:4.0f}%) {rate:4.1f}it/s ETA {eta:4.0f}s")
        else:
            msg = f"\r{desc}: {count}건 {rate:4.1f}it/s 경과 {elapsed:4.0f}s"
        pad = " " * max(0, last_len - len(msg))
        sys.stdout.write(msg + pad)
        sys.stdout.flush()
        last_len = len(msg)
    sys.stdout.write("\n")
    sys.stdout.flush()


def progress(iterable, total=None, desc="진행"):
    """진행 표시 래퍼. tqdm이 있으면 그걸, 없으면 폴백을 사용."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc,
                    ncols=88, dynamic_ncols=False)
    except ImportError:
        return _fallback_progress(iterable, total=total, desc=desc)