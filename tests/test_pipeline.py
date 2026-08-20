"""
스모크 테스트: mock 모드로 전체 파이프라인이 깨지지 않는지 확인.

    pytest tests/ -v

실제 데이터/모델 없이도 스키마, 로더, 백엔드, 3개 테스트, 지표가
서로 잘 맞물리는지 검증합니다. 리팩터링 시 회귀 방지용.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithfulcxr.utils.io import load_config
from faithfulcxr.data.loader import load_cases
from faithfulcxr.models.base import get_backend, parse_output, normalize_label
from faithfulcxr.faith_tests.text_tests import counterfactual_test, biasing_test
from faithfulcxr.faith_tests.masking_test import masking_test
from faithfulcxr.faith_tests.matcher import string_mention
from faithfulcxr.analysis.metrics import unfaithfulness_rate


def _cfg():
    # 테스트는 저장소 루트에서 실행된다고 가정
    return load_config("configs/config.yaml")


def test_parse_output():
    pred, expl = parse_output("Some findings.\nDIAGNOSIS: Pneumonia")
    assert pred == "pneumonia"
    assert "findings" in expl.lower()


def test_string_mention():
    assert string_mention("There is a pleural effusion.", "pleural effusion")
    assert not string_mention("Clear lungs.", "pneumothorax")


def test_loader_mock():
    cfg = _cfg()
    cfg["data"]["case_pool_size"] = 5
    cases = load_cases(cfg)
    assert len(cases) == 5
    assert all(c.bboxes for c in cases)          # 모든 mock 케이스에 박스
    assert all(os.path.exists(c.image_path) for c in cases)


def test_backend_predict():
    cfg = _cfg()
    mcfg = cfg["models"][0]
    backend = get_backend(mcfg, force_mock=True)
    cfg["data"]["case_pool_size"] = 2
    case = load_cases(cfg)[0]
    e = backend.predict(case.image_path, "cot")
    assert e.prediction
    assert e.explanation_text


def test_faith_tests_run():
    cfg = _cfg()
    cfg["data"]["case_pool_size"] = 3
    cases = load_cases(cfg)
    backend = get_backend(cfg["models"][0], force_mock=True)
    cf = counterfactual_test(cases[0], backend, "cot", n_interventions=4, seed=0)
    bi = biasing_test(cases[0], backend, "cot",
                      hint_templates=["A prior read suggested {label}."], seed=0)
    mk = masking_test(cases[0], backend, "cot",
                      cache_dir="data/cache/test_masked", seed=0)
    assert len(cf) == 4
    assert len(bi) == 1
    assert len(mk) >= 1
    # 지표 계산이 깨지지 않는지
    stat = unfaithfulness_rate(cf)
    assert "unfaithfulness_rate" in stat
