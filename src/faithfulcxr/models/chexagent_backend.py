"""
CheXagent-2 전용 백엔드.

CheXagent-2-3b는 MedGemma와 API가 다릅니다:
  - AutoModelForCausalLM + AutoTokenizer + AutoProcessor
  - trust_remote_code=True 필수
  - 커스텀 프롬프트 형식: " USER: <s>{prompt} ASSISTANT: <s>"
  - transformers==4.40.0 권장
주로 Colab T4에서 실행 (bfloat16, ~7GB).
"""
from __future__ import annotations

from faithfulcxr.models.base import VLMBackend


class CheXagentBackend(VLMBackend):
    def __init__(self, model_id: str, spec: dict):
        super().__init__(model_id, spec)
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.processor = None

    def _lazy_load(self):
        if self._loaded:
            return
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  AutoProcessor)

        hf_name = self.spec["hf_name"]
        print(f"[chexagent] '{hf_name}' 로드 중...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            hf_name, trust_remote_code=True)

        load_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if self.spec.get("quantization") == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kwargs)
        self.model.eval()
        self._loaded = True
        print(f"[chexagent] 준비 완료")

    def _generate(self, image_path: str, prompt: str) -> str:
        self._lazy_load()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        text = f" USER: <s>{prompt} ASSISTANT: <s>"
        inputs = self.processor(
            images=[image], text=text, return_tensors="pt",
        ).to(self.model.device)
        if self.spec.get("quantization") != "4bit":
            inputs = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v)
                      for k, v in inputs.items()}

        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=256,
                                      do_sample=False)
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        gen = out[0][input_len:]
        text_out = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text_out.strip()