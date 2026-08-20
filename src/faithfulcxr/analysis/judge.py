"""
LLM-as-judge 유틸.

설명의 '그럴듯함(plausibility)'을 LLM이 1~5점으로 평가합니다.
충실성과는 다른 축입니다:
  - 그럴듯함 = 방사선의가 보기에 설명이 말이 되는가 (설득력)
  - 충실성  = 설명이 모델의 실제 판단 근거를 반영하는가
핵심 가설(RQ2): 그럴듯함이 높아도 충실성은 낮을 수 있다.
"""
from __future__ import annotations

import os
import re


def make_openai_judge(model: str = "gpt-5.4"):
    """OpenAI 텍스트 전용 judge 함수 생성. judge_fn(prompt)->str."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def judge_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=64,
        )
        return resp.choices[0].message.content or ""

    return judge_fn

def make_gemini_judge(model: str = "gemini-2.5-flash"):
    """Google Gemini 텍스트 전용 judge. GOOGLE_API_KEY 필요."""
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def judge_fn(prompt: str) -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    return judge_fn


def make_anthropic_judge(model: str = "claude-sonnet-5"):
    """Anthropic Claude 텍스트 전용 judge. ANTHROPIC_API_KEY 필요."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def judge_fn(prompt: str) -> str:
        resp = client.messages.create(
            model=model, max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))

    return judge_fn


def make_judge(provider: str, model: str = None):
    """provider별 judge 통합 팩토리. openai|gemini|anthropic."""
    defaults = {
        "openai": "gpt-5.4",
        "gemini": "gemini-2.5-flash",
        "anthropic": "claude-sonnet-5",
    }
    m = model or defaults[provider]
    if provider == "openai":
        return make_openai_judge(m)
    if provider == "gemini":
        return make_gemini_judge(m)
    if provider == "anthropic":
        return make_anthropic_judge(m)
    raise ValueError(f"알 수 없는 judge provider: {provider}")


PLAUSIBILITY_RUBRIC = (
    "You are an expert radiologist evaluating an AI-generated explanation for a "
    "chest X-ray reading. Rate ONLY how plausible/convincing the explanation is "
    "to a radiologist — i.e., whether it reads as clinically sound and coherent. "
    "Do NOT judge whether the final diagnosis is correct; judge the quality and "
    "plausibility of the reasoning itself.\n\n"
    "Use this 1-5 scale:\n"
    "1 = incoherent or clinically nonsensical\n"
    "2 = weak, vague, or partly implausible\n"
    "3 = acceptable but generic\n"
    "4 = clinically sound and specific\n"
    "5 = expert-level, precise, and well-justified\n\n"
    "Explanation to rate:\n\"\"\"\n{explanation}\n\"\"\"\n\n"
    "Respond with ONLY a single integer from 1 to 5."
)


def score_plausibility(explanation_text: str, judge_fn, n_repeats: int = 3) -> float:
    """설명 하나를 n_repeats회 평가해 평균 점수 반환. 전부 실패면 nan."""
    prompt = PLAUSIBILITY_RUBRIC.format(explanation=explanation_text)
    scores = []
    for _ in range(n_repeats):
        raw = judge_fn(prompt)
        m = re.search(r"[1-5]", raw)
        if m:
            scores.append(int(m.group()))
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)