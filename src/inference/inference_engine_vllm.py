import os

import time
import yaml
from typing import Literal
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


CONFIG_INFER_FILE = "configs/model_inference_schema.yaml"


class VLLMEngine:
    """
    vLLM inference engine, config-driven, supports LLM/VLM; base/SFT/LoRA/RL model types
    
    Core interface:
        engine = VLLMEngine("qwen3_4b_i")
        
        # LLM: single-turn QA
        answer = engine.ask("What is the SMILES of water?")
        
        # VLM: single-turn QA (text-only)
        answer = engine.ask("Describe this image.")
        
        # VLM: single-turn image-text (OCR task)
        answer = engine.ask_with_image("What molecules are in this image?", pil_image)
        
        # Inference from dataset sample
        for sample in dataset:
            answer = engine.chat_from_sample(sample, prompt_key="question")
        
        # Full control: custom messages (batch supported)
        outputs = engine.chat([
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "What is AI?"}],
        ])
    """
    
    def __init__(self, name: str, config_path: str = CONFIG_INFER_FILE):
        self.name = name
        self.config = self._load_config(name, config_path)
        self.modality: Literal["llm", "vlm"] = self.config.get("modality", "llm")
        self.llm: LLM | None = None
        self._lora_request: LoRARequest | None = None
        
    def _load_config(self, name: str, config_path: str) -> dict:
        with open(config_path, "r") as f:
            all_config = yaml.safe_load(f)
        
        vllm_config = all_config.get("vllm", {})
        model_config = vllm_config.get(name)
        if not model_config:
            available = list(vllm_config.keys())
            raise ValueError(f"Model '{name}' not found in config. Available: {available}")
        return model_config
    
    def _get_default_sampling_params(self, **overrides) -> SamplingParams:
        defaults = self.config.get("sampling_params", {})
        params = {**defaults, **overrides}
        return SamplingParams(**params)
    
    # ==================== Build ====================
    
    def build(self) -> LLM:
        model_type = self.config.get("type", "base")
        
        ### TODO: maybe only keep base/lora, sft and rl after merged can be base type
        build_methods = {
            "base": self._build_base,
            "sft": self._build_sft,
            "lora": self._build_lora,
            "rl": self._build_rl,
        }
        
        if model_type not in build_methods:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.llm = build_methods[model_type]()
        return self.llm
    
    def _build_base(self) -> LLM:
        return LLM(
            **self.config["vllm_init"],
            **self.config.get("vllm_extra", {}),
        )
    
    def _build_sft(self) -> LLM:
        return self._build_base()
    
    def _build_lora(self) -> LLM:
        lora_cfg = self.config["vllm_lora_request"]
        self._lora_request = LoRARequest(
            lora_name=lora_cfg["name"],
            lora_int_id=1,
            lora_path=lora_cfg["path"],
        )

        print(f"✅ LoRA request built: {self._lora_request=}")
        
        return LLM(
            **self.config["vllm_init"],
            **self.config.get("vllm_extra", {}),
        )
    
    def _build_rl(self) -> LLM:
        return self._build_base()
    
    # ==================== Chat ====================
    
    def chat(
        self,
        messages: list[dict] | list[list[dict]],
        sampling_params: SamplingParams | None = None,
        lora_request: LoRARequest | None = None,
        return_text: bool = True,
    ) -> list:
        """
        Args:
            messages: OpenAI-format message list
                - Single: [{"role": "user", "content": "question"}]
                - Batch: [[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]]
                - VLM with image: [{"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "question"}
                  ]}]
            sampling_params: Custom sampling params, None uses config defaults
            lora_request: LoRA request, None uses config default or no LoRA; overrides config LoRA if provided
            return_text: Whether to return only generated text instead of full RequestOutput

        Returns:
            List of vLLM RequestOutput objects
        """
        if self.llm is None:
            self.build()
        
        if sampling_params is None:
            sampling_params = self._get_default_sampling_params()
        
        lora_kwargs = {"lora_request": lora_request} if lora_request else {"lora_request": self._lora_request} if self._lora_request else {}
        
        outputs = self.llm.chat(
            messages,
            sampling_params=sampling_params,
            **lora_kwargs,
        )
        return [output.outputs[0].text if return_text else output for output in outputs]
    
    # ==================== Convenience Methods ====================
    
    def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        **sampling_overrides,
    ) -> str:
        """
        Single-turn QA (text-only), works for both LLM and VLM.
        
        Args:
            question: User question
            system_prompt: Optional system prompt
            **sampling_overrides: Override default sampling params
            
        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        sampling_params = self._get_default_sampling_params(**sampling_overrides)
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=True)
        return outputs[0]
    
    def ask_with_image(
        self,
        question: str,
        image: Image.Image,
        system_prompt: str | None = None,
        **sampling_overrides,
    ) -> str:
        """
        Single-turn image-text QA, VLM only.
        
        Args:
            question: User question
            image: PIL Image object
            system_prompt: Optional system prompt
            **sampling_overrides: Override default sampling params
            
        Returns:
            Generated text
        """
        if self.modality != "vlm":
            raise ValueError(
                f"ask_with_image is only available for VLM models. "
                f"Current model '{self.name}' is '{self.modality}'."
            )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": question},
            ],
        })
        
        sampling_params = self._get_default_sampling_params(**sampling_overrides)
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=True)
        return outputs[0]
    
    def chat_from_sample(
        self,
        sample: dict,
        prompt_key: str = "prompt",
        image_key: str | None = "input_rep",
        system_prompt: str | None = None,
        **sampling_overrides,
    ) -> str:
        """
        Construct messages and infer from a single HF dataset sample.
        
        Args:
            sample: Single HF dataset sample (dict)
            prompt_key: Key for prompt field
            image_key: Key for image field, None for text-only; should be PIL Image in sample
            system_prompt: Optional system prompt
            **sampling_overrides: Override default sampling params
            
        Returns:
            Generated text
        """
        question = sample[prompt_key]
        image = sample.get(image_key) if image_key else None
        
        if image is not None:
            return self.ask_with_image(question, image, system_prompt, **sampling_overrides)
        else:
            return self.ask(question, system_prompt, **sampling_overrides)
    
    # ==================== Debug ====================
    
    def debug(
        self,
        question: str,
        image: Image.Image | None = None,
        system_prompt: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Debug interface, supports text-only and image-text.
        
        Args:
            question: User question
            image: Optional PIL Image (for VLM)
            system_prompt: Optional system prompt
            verbose: Whether to print debug info
            
        Returns:
            Dictionary containing answer, messages, sampling_params, timing, etc.
        """
        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if image is not None:
            if self.modality != "vlm":
                raise ValueError(f"Image input requires VLM model, got '{self.modality}'.")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": question},
                ],
            })
        else:
            messages.append({"role": "user", "content": question})
        
        sampling_params = self._get_default_sampling_params()
        
        if verbose:
            print("=" * 60)
            print(f"[DEBUG] Model: {self.name} ({self.modality})")
            print(f"[DEBUG] Messages:")
            for msg in messages:
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image_pil":
                            print(f"  - [Image: {item['image_pil'].size}]")
                        else:
                            print(f"  - [{item['type']}]: {item['text']}")
                else:
                    print(f"  [{msg['role']}]: {content}")
            print(f"[DEBUG] Sampling params: {sampling_params}")
            print("-" * 60)
        
        # Timed inference
        start = time.time()
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=False)
        elapsed = time.time() - start
        
        result = outputs[0]
        answer = result.outputs[0].text
        prompt_tokens = len(result.prompt_token_ids) if hasattr(result, 'prompt_token_ids') else 0
        output_tokens = len(result.outputs[0].token_ids) if result.outputs[0].token_ids else 0
        
        if verbose:
            print(f"[DEBUG] Answer: {answer!r}")
            print(f"[DEBUG] Tokens: prompt={prompt_tokens}, output={output_tokens}")
            print(f"[DEBUG] Time: {elapsed:.2f}s")
            print("=" * 60)
        
        return {
            "answer": answer,
            "messages": messages,
            "sampling_params": vars(sampling_params),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "time": elapsed,
        }


if __name__ == "__main__":
    # # === LLM usage ===
    # print("=== LLM Example ===")
    # llm_engine = VLLMEngine("qwen3_4b_i")
    # answer = llm_engine.ask("What is the SMILES of water?")
    # print(f"Answer: {answer}")
    
    # === VLM usage ===
    print("\n=== VLM Example ===")
    vlm_engine = VLLMEngine("qwen3_vl_4b_i")
    
    # Text-only QA
    answer = vlm_engine.ask("What is a molecule?")
    print(f"Answer: {answer}")
    
    # Image-text QA (OCR task)
    from PIL import Image
    from urllib.request import urlopen

    img = Image.open(urlopen("https://picsum.photos/300"))
    answer = vlm_engine.ask_with_image("What is in this image?", img)
    print(f"Answer: {answer}")
    
    # Inference from dataset sample
    # from datasets import load_dataset
    sample = {
        "question": "Describe this image.",
        "input_rep": img,
    }
    answer = vlm_engine.chat_from_sample(sample, prompt_key="question")
    print(answer)
    
    # # Batch inference
    # all_messages = [
    #     [{"role": "user", "content": item["question"]}]
    #     for item in ds["test"].select(range(100))
    # ]
    # outputs = vlm_engine.chat(all_messages)
    
    # Debug
    result = vlm_engine.debug("Describe this image.", image=img)