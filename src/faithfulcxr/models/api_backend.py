"""
API 백엔드 (GPT-5.4 등 OpenAI 최신 멀티모달).

⚠️ 데이터 정책: MIMIC 데이터를 API로 보낼 땐 zero-data-retention 확인 필수.
   VinDr(공개)에서는 무관.

API 키: 환경변수 OPENAI_API_KEY

최신 모델(GPT-5 계열) 주의:
    - max_tokens 대신 max_completion_tokens 사용
    - temperature 생략 (reasoning 모델 특성)
    - 재현성 위해 스냅샷 고정 권장: gpt-5.4-2026-03-05
"""
from __future__ import annotations

import base64
import os

from faithfulcxr.models.base import VLMBackend


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _media_type(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if p.endswith(".webp"):
        return "image/webp"
    return "image/png"


class APIBackend(VLMBackend):
    def __init__(self, model_id: str, spec: dict):
        super().__init__(model_id, spec)
        self.provider = spec.get("api_provider", "openai")
        self.api_model = spec.get("api_model", "gpt-5.4")

    def _generate(self, image_path: str, prompt: str) -> str:
        if self.provider == "openai":
            return self._openai(image_path, prompt)
        raise NotImplementedError(f"프로바이더 미구현: {self.provider}")

    def _openai(self, image_path: str, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        b64 = _b64_image(image_path)
        media = _media_type(image_path)
        resp = client.chat.completions.create(
            model=self.api_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media};base64,{b64}"}},
                ],
            }],
            max_completion_tokens=256,
        )
        return resp.choices[0].message.content or ""