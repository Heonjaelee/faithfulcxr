"""
HuggingFace 로컬 백엔드 (4-bit 양자화, RTX 3050용).

MedGemma-4B(google/medgemma-4b-it)를 4~6GB VRAM에서 돌리기 위한 설정.
모델 카드 기준 정확한 클래스(AutoModelForImageTextToText)와 chat template을 사용.

사전 준비 (MedGemma는 게이트 모델):
    1) https://huggingface.co/google/medgemma-4b-it 에서 "Acknowledge license" 클릭
    2) pip install -U transformers accelerate bitsandbytes
    3) huggingface-cli login  (토큰 입력)
"""
from __future__ import annotations

from faithfulcxr.models.base import VLMBackend


class HFBackend(VLMBackend):
    def __init__(self, model_id: str, spec: dict):
        super().__init__(model_id, spec)
        self._loaded = False
        self.model = None
        self.processor = None

    def _lazy_load(self):
        if self._loaded:
            return
        import torch
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  BitsAndBytesConfig)

        hf_name = self.spec["hf_name"]
        print(f"[hf] '{hf_name}' 로드 중... 첫 실행은 다운로드로 오래 걸립니다(~8GB).")

        quant_cfg = None
        load_kwargs = {"device_map": "auto"}
        if self.spec.get("quantization") == "4bit":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = quant_cfg
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(hf_name)
        self.model = AutoModelForImageTextToText.from_pretrained(hf_name, **load_kwargs)
        self.model.eval()
        self._loaded = True
        print(f"[hf] '{hf_name}' 준비 완료")

    def _generate(self, image_path: str, prompt: str) -> str:
        self._lazy_load()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = gen[0][input_len:]
        text = self.processor.decode(gen, skip_special_tokens=True)
        return text.strip()