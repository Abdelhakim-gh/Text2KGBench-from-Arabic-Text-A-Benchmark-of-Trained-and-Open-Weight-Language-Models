"""Shared model-calling interface. Ollama today; the interface is written so
an HF/transformers runner can be added later (for a study that needs a model
Ollama doesn't serve) without changing any caller.

`think=False` is required, not cosmetic: qwen3.5 defaults to emitting a
separate `thinking` field and, if left on, can burn the entire token budget
on it before producing any `message.content`, leaving the answer empty even
though the call "succeeded." Confirmed a safe no-op on non-reasoning models.

`num_ctx` must be passed explicitly on every call, not left to the model's
Modelfile default. Confirmed by direct testing (the AceGPT-13B finding in
configs/llm_models.yaml): a Modelfile's `PARAMETER num_ctx 8192` is not enough on
its own. Ollama silently falls back to 2048 tokens whenever the API request
omits `options.num_ctx`, regardless of what the Modelfile declares, and
raises `exceed_context_size_error` on any prompt longer than that with no
warning beforehand. Since this ontology-guided prompt (a 20-relation list
plus an Arabic sentence) can run past 2048 tokens for longer sentences, this
was silently truncating or failing calls for every model, not just the one
where it happened to be first noticed.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import ollama

logger = logging.getLogger(__name__)

DEFAULT_NUM_CTX = 8192  # matches the context length most of the roster already reports via `ollama ps`


@dataclass
class GenerationResult:
    text: str
    tokens_in: int
    tokens_out: int


class ModelRunner(ABC):
    @abstractmethod
    def generate(self, prompt: str, seed: int, temperature: float, max_tokens: int) -> GenerationResult: ...


class OllamaRunner(ModelRunner):
    def __init__(self, model_name: str, host: str = "http://localhost:11434", num_ctx: int = DEFAULT_NUM_CTX):
        self.model_name = model_name
        self.client = ollama.Client(host=host)
        self.num_ctx = num_ctx

    def generate(self, prompt: str, seed: int, temperature: float, max_tokens: int) -> GenerationResult:
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "seed": seed, "num_predict": max_tokens, "num_ctx": self.num_ctx},
            think=False,
        )
        text = response.message.content or ""
        return GenerationResult(
            text=text,
            tokens_in=response.prompt_eval_count or 0,
            tokens_out=response.eval_count or 0,
        )

    def ping(self) -> bool:
        """One trivial call to confirm connectivity and correct Arabic round-tripping."""
        result = self.generate("اكتب كلمة: مرحبا", seed=1, temperature=0.0, max_tokens=16)
        ok = len(result.text.strip()) > 0
        logger.info("%s: ping %s, response=%r", self.model_name, "OK" if ok else "EMPTY", result.text)
        return ok
