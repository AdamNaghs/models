from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fineweb_gpt_common import (  # noqa: E402
    ASST_PREFIX,
    GPT,
    TURN_SUFFIX,
    USER_PREFIX,
    resolve_tokenizer_path,
    tokenizer_fingerprint,
)


@dataclass
class ChoiceScore:
    text: str
    raw_logprob: float
    normalized_logprob: float
    token_count: int


class FineWebGPTAdapter:
    def __init__(self, ckpt_path: str, tok_path: str | None = None, device: str | None = None):
        self.ckpt_path = str(Path(ckpt_path).resolve())
        self.tok_path = resolve_tokenizer_path(self.ckpt_path, tok_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.sp = spm.SentencePieceProcessor(model_file=self.tok_path)
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = self.ckpt["args"]
        self.context = int(self.cfg["context"])

        ckpt_tok_fp = self.ckpt.get("tokenizer_fingerprint")
        tok_fp = tokenizer_fingerprint(self.sp)
        if ckpt_tok_fp and ckpt_tok_fp != tok_fp:
            raise ValueError(
                "Tokenizer mismatch: tokenizer.model does not match checkpoint tokenizer. "
                "Use the same tokenizer from training/finetuning."
            )

        self.is_finetuned = "chat_format" in self.ckpt
        self.chat_format = self.ckpt.get(
            "chat_format",
            {"user_prefix": USER_PREFIX, "asst_prefix": ASST_PREFIX, "turn_suffix": TURN_SUFFIX},
        )

        self.model = GPT(
            vocab=self.ckpt["vocab"],
            context=self.cfg["context"],
            n_embd=self.cfg["n_embd"],
            n_head=self.cfg["n_head"],
            n_layer=self.cfg["n_layer"],
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.model.eval()

        eos_id = self.sp.eos_id()
        self.stop_tokens = {eos_id} if eos_id >= 0 else set()
        self.stop_sequences = [
            self.sp.encode(self.chat_format["user_prefix"].strip(), out_type=int),
            self.sp.encode(self.chat_format["asst_prefix"].strip(), out_type=int),
        ]

    def encode(self, text: str) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if not ids and self.sp.bos_id() >= 0:
            return [self.sp.bos_id()]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def _truncate_for_context(self, prompt_ids: list[int], continuation_ids: list[int]) -> tuple[list[int], int]:
        max_total = self.context + 1
        full_ids = prompt_ids + continuation_ids
        continuation_start = len(prompt_ids)

        if len(full_ids) <= max_total:
            return full_ids, continuation_start

        excess = len(full_ids) - max_total
        if excess < len(prompt_ids):
            prompt_ids = prompt_ids[excess:]
            full_ids = prompt_ids + continuation_ids
            continuation_start = len(prompt_ids)
            return full_ids, continuation_start

        full_ids = full_ids[-max_total:]
        continuation_start = max(0, max_total - len(continuation_ids))
        return full_ids, continuation_start

    @torch.no_grad()
    def score_continuation(self, prompt: str, continuation: str) -> ChoiceScore:
        prompt_ids = self.encode(prompt)
        continuation_ids = self.encode(continuation)
        if not continuation_ids:
            return ChoiceScore(text=continuation, raw_logprob=float("-inf"), normalized_logprob=float("-inf"), token_count=0)

        full_ids, continuation_start = self._truncate_for_context(prompt_ids, continuation_ids)
        if len(full_ids) < 2:
            return ChoiceScore(text=continuation, raw_logprob=float("-inf"), normalized_logprob=float("-inf"), token_count=0)

        inputs = torch.tensor(full_ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        targets = torch.tensor(full_ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)
        logits, _ = self.model(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)

        start_idx = max(0, continuation_start - 1)
        continuation_token_log_probs = token_log_probs[start_idx:]
        raw = float(continuation_token_log_probs.sum().item())
        token_count = int(continuation_token_log_probs.numel())
        normalized = raw / max(token_count, 1)
        return ChoiceScore(
            text=continuation,
            raw_logprob=raw,
            normalized_logprob=normalized,
            token_count=token_count,
        )

    @torch.no_grad()
    def score_choices(self, prompt: str, choices: list[str], metric: str = "normalized") -> tuple[int, list[ChoiceScore]]:
        scores = [self.score_continuation(prompt, choice) for choice in choices]
        if metric == "raw":
            key_fn = lambda s: s.raw_logprob
        else:
            key_fn = lambda s: s.normalized_logprob
        best_idx = max(range(len(scores)), key=lambda i: key_fn(scores[i]))
        return best_idx, scores

    def format_messages(self, messages: list[dict]) -> str:
        user_prefix = self.chat_format["user_prefix"]
        asst_prefix = self.chat_format["asst_prefix"]
        turn_suffix = self.chat_format["turn_suffix"]
        prompt = ""
        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                prompt += user_prefix + content + turn_suffix
            elif role == "assistant":
                prompt += asst_prefix + content + turn_suffix
        prompt += asst_prefix
        return prompt

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 256, temp: float = 0.7, top_p: float = 0.9) -> str:
        prompt_ids = self.encode(prompt)
        if len(prompt_ids) > self.context - max_tokens:
            prompt_ids = prompt_ids[-(self.context - max_tokens):]
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            stop_tokens=self.stop_tokens,
            stop_sequences=self.stop_sequences,
        )
        generated = out[0].tolist()[len(prompt_ids):]
        text = self.decode(generated).strip()
        for marker in [self.chat_format["user_prefix"].strip(), self.chat_format["asst_prefix"].strip(), "###"]:
            if marker and marker in text:
                text = text[: text.index(marker)].strip()
        return text.replace("</s>", "").strip()

    def generate_from_messages(self, messages: list[dict], max_tokens: int = 256, temp: float = 0.7, top_p: float = 0.9) -> str:
        if self.is_finetuned:
            return self.generate(self.format_messages(messages), max_tokens=max_tokens, temp=temp, top_p=top_p)
        transcript = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = (msg.get("content") or "").strip()
            if content:
                transcript.append(f"{role}: {content}")
        transcript.append("Assistant:")
        return self.generate("\n".join(transcript), max_tokens=max_tokens, temp=temp, top_p=top_p)

    @torch.no_grad()
    def perplexity_from_text(self, text: str, stride: int | None = None) -> dict:
        ids = self.encode(text)
        if len(ids) < 2:
            return {"token_count": 0, "avg_nll": math.nan, "perplexity": math.nan}

        stride = stride or max(1, self.context // 2)
        total_nll = 0.0
        total_tokens = 0

        for start in range(0, len(ids) - 1, stride):
            full = ids[start:start + self.context + 1]
            if len(full) < 2:
                break
            inputs = torch.tensor(full[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            targets = torch.tensor(full[1:], dtype=torch.long, device=self.device).unsqueeze(0)
            logits, _ = self.model(inputs)
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll += float(nll)
            total_tokens += int(targets.numel())
            if start + self.context + 1 >= len(ids):
                break

        avg_nll = total_nll / max(total_tokens, 1)
        return {
            "token_count": total_tokens,
            "avg_nll": avg_nll,
            "perplexity": math.exp(avg_nll) if total_tokens else math.nan,
        }
