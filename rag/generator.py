"""Answer generation with a local instruction-tuned seq2seq model (FLAN-T5)."""
from typing import Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import Config

TEMPLATE = (
    "Answer the question using only the context below. Give a short, direct answer. "
    "If the context does not contain the answer, reply exactly: I don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class Generator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.gen_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.gen_model).to(self.device)
        self.model.eval()

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """Use as many retrieved chunks as fit in the token budget (best-ranked first)."""
        for n in range(len(contexts), 0, -1):
            context = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(contexts[:n]))
            prompt = TEMPLATE.format(context=context, question=question)
            if len(self.tokenizer(prompt).input_ids) <= self.cfg.max_input_tokens:
                return prompt
        return TEMPLATE.format(context=contexts[0], question=question)

    @torch.no_grad()
    def generate(self, question: str, hits: List[Dict]) -> str:
        prompt = self.build_prompt(question, [h["text"] for h in hits])
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_input_tokens + 24
        ).to(self.device)
        output = self.model.generate(
            **inputs, max_new_tokens=self.cfg.max_new_tokens, num_beams=self.cfg.num_beams
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
