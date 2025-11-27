import torch
from typing import Optional


class HfLocalCaller:
    """
    Minimal wrapper around a local HF model (LLaMA, Mistral, etc.)
    to expose a `.call(prompt: str) -> str` interface.
    """

    def __init__(
        self,
        tokenizer,
        model,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

        self.base_max_new_tokens = max_new_tokens
        self.base_temperature = temperature
        self.base_top_p = top_p

        # Make sure padding is defined
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a continuation for `prompt` and return the decoded text.
        """
        temp = self.base_temperature if temperature is None else temperature
        mnt = self.base_max_new_tokens if max_new_tokens is None else max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = dict(
            max_new_tokens=mnt,
            do_sample=(temp > 0),
            temperature=max(temp, 1e-5),
            top_p=self.base_top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
