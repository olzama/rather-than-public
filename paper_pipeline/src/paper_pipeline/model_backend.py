"""Runs chat-style prompts through either the OpenAI API or a local transformers model.

Generation and revision share one backend instance so a local model is loaded
into memory (and onto the GPU) exactly once per pipeline run. Review may use a
different model and therefore a separate backend, even though each stage still
uses its own decoding parameters and system prompt.
"""

import logging
import os
from typing import Tuple

from .config_loader import ChatCfg, DecodingCfg, ModelCfg, QuantizationCfg
from .prompts import is_openai_model
from .utils import estimate_cost

logger = logging.getLogger(__name__)


class ChatModelBackend:
    def __init__(self, model_cfg: ModelCfg, quant_cfg: QuantizationCfg):
        self.model_cfg = model_cfg
        self.quant_cfg = quant_cfg
        self.is_openai = is_openai_model(model_cfg.name)

        if self.is_openai:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("Using OpenAI API model: %s", model_cfg.name)
        else:
            self._load_local_model()

    def _load_local_model(self) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self._torch = torch
        mode = self.quant_cfg.mode.lower()
        cfg = AutoConfig.from_pretrained(self.model_cfg.name)
        prequant = getattr(cfg, "quantization_config", None)

        quantization_config = None
        if prequant is None:
            if mode == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.quant_cfg.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.quant_cfg.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=getattr(
                        torch, self.quant_cfg.bnb_4bit_compute_dtype
                    ),
                )
            elif mode == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=self.quant_cfg.llm_int8_threshold,
                    llm_int8_has_fp16_weight=self.quant_cfg.llm_int8_has_fp16_weight,
                    llm_int8_enable_fp32_cpu_offload=self.quant_cfg.llm_int8_enable_fp32_cpu_offload,
                )
            else:
                assert mode == "none", (
                    f"Quantization mode '{mode}' not supported. "
                    "Use '4bit', '8bit' or 'none'."
                )

        load_kwargs = {
            "device_map": self.model_cfg.device,
            "dtype": self.model_cfg.dtype,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        logger.info("Loading local model: %s", self.model_cfg.name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.name, **load_kwargs
        ).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.name)
        if (
            self._tokenizer.pad_token_id is None
            and self._tokenizer.eos_token_id is not None
        ):
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self, prompt: str, decoding_cfg: DecodingCfg, chat_cfg: ChatCfg
    ) -> Tuple[str, float]:
        """Run one prompt to completion. Returns (text, estimated_cost_usd)."""
        if self.is_openai:
            return self._generate_openai(prompt, decoding_cfg, chat_cfg)
        return self._generate_local(prompt, decoding_cfg, chat_cfg)

    def _generate_openai(
        self, prompt: str, decoding_cfg: DecodingCfg, chat_cfg: ChatCfg
    ) -> Tuple[str, float]:
        if self.model_cfg.name.startswith("gpt-6") or self.model_cfg.name.startswith("gpt-5"):
            response = self._client.chat.completions.create(
                model=self.model_cfg.name,
                max_completion_tokens=decoding_cfg.max_completion_tokens,
                messages=[
                    {"role": "system", "content": chat_cfg.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            response = self._client.chat.completions.create(
                model=self.model_cfg.name,
                temperature=decoding_cfg.temperature,
                top_p=decoding_cfg.top_p,
                max_tokens=decoding_cfg.max_new_tokens,
                messages=[
                    {"role": "system", "content": chat_cfg.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

        if response.choices[0].finish_reason == "length":
            raise ValueError("Response was truncated; increase decoding.max_new_tokens")
        text = (response.choices[0].message.content or "").strip()
        cost = estimate_cost(response.usage, self.model_cfg.name)
        return text, cost

    def _build_messages(self, user_prompt: str, chat_cfg: ChatCfg):
        is_chat = self.model_cfg.is_chat_model
        has_template = bool(getattr(self._tokenizer, "chat_template", None))
        sys_prompt = (chat_cfg.system_prompt or "").strip()

        if not is_chat:
            return user_prompt

        if not has_template:
            return f"{sys_prompt}\n\n{user_prompt}" if sys_prompt else user_prompt

        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        return self._tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )

    def _generate_local(
        self, prompt: str, decoding_cfg: DecodingCfg, chat_cfg: ChatCfg
    ) -> Tuple[str, float]:
        message = self._build_messages(prompt, chat_cfg)
        inputs = self._tokenizer(message, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        d = decoding_cfg

        with self._torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=d.max_new_tokens,
                do_sample=d.do_sample,
                temperature=d.temperature,
                top_p=d.top_p,
                top_k=d.top_k,
                repetition_penalty=d.repetition_penalty,
                num_beams=d.num_beams,
                num_return_sequences=d.num_return_sequences,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        gen_only = out.sequences[:, inputs["input_ids"].shape[1] :]
        decoded = self._tokenizer.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        text = "\n\n".join(t.strip() for t in decoded if t is not None)
        return text, 0.0
