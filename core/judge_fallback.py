"""LLM-as-Judge fallback for unparseable model outputs (paper §3 Eval Step 2).

Pipeline (per parse pass):
    extract-match (Evaluator.parse_*_output)  →  if 'UNPARSEABLE' fall back to LLMJudge

The judge re-asks a small auxiliary model to map the original sample's raw
response onto one of the allowed labels. Defaults to gemini-2.5-flash via the
Google OpenAI-compatible endpoint. If no api key is configured, the judge is
disabled and unparseable predictions stay 'UNPARSEABLE'.

Config (configs/experiment.yaml under ab_experiment.judge):
    enabled: true
    model_name: gemini-2.5-flash
    api_key:    null         # falls back to env GEMINI_API_KEY / GOOGLE_API_KEY
    base_url:   null         # falls back to Google's OpenAI-compatible endpoint
    max_workers: 5
"""
import asyncio
import os
from typing import List, Optional

import openai

from .config_loader import get_block


# Google AI Studio's OpenAI-compatible chat-completion endpoint.
# Docs: https://ai.google.dev/gemini-api/docs/openai
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class LLMJudge:
    """Async judge client. One instance is shared across all judge calls in a run."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_workers: int = 5,
    ):
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        # Resolve API key: explicit arg > GEMINI_API_KEY > GOOGLE_API_KEY.
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "LLMJudge: no API key. Provide via ab_experiment.judge.api_key "
                "in config, or set GEMINI_API_KEY / GOOGLE_API_KEY in the env."
            )
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url or DEFAULT_GEMINI_BASE_URL,
        )
        self.semaphore = asyncio.Semaphore(max_workers)

    # ------------------------------------------------------------------
    # Internal: single async call
    # ------------------------------------------------------------------
    async def _query(self, content: str) -> str:
        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[LLMJudge] error: {e}")
                return ""

    # ------------------------------------------------------------------
    # MCQ judge — letter-coded
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mcq_prompt(raw_text: str, options: List[str], with_unknown: bool) -> str:
        letters = "ABCDEF"
        opts_lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
        if with_unknown:
            opts_lines.append(f"{letters[len(options)]}. Unknown")
        valid = letters[: len(options) + (1 if with_unknown else 0)]
        return (
            "You are an answer-key adjudicator. A model was asked a multiple-choice\n"
            "question with the following options:\n\n"
            f"{chr(10).join(opts_lines)}\n\n"
            "The model produced this raw response:\n"
            f"---\n{raw_text}\n---\n\n"
            f"Which option did the model intend to choose? Reply with a single\n"
            f"letter from {{ {' / '.join(valid)} }}. If the model's intent is\n"
            f"genuinely unclear, reply with the literal token 'UNCLEAR'."
        )

    async def judge_mcq(self, raw_text: str, options: List[str],
                         with_unknown: bool) -> str:
        """Re-ask the judge to map a raw model output to a single option letter.

        Returns the judge's raw text reply; the caller re-parses through
        Evaluator.parse_mcq_output to turn it into 'A'/'B'/.../'UNKNOWN'/'UNPARSEABLE'.
        """
        return await self._query(self._build_mcq_prompt(raw_text, options, with_unknown))

    # ------------------------------------------------------------------
    # Judge (T/F + Unknown) judge — verb-coded via LabelScheme
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tf_prompt(raw_text: str, scheme, with_unknown: bool) -> str:
        verbs = [scheme.pos_verb, scheme.neg_verb]
        if with_unknown:
            verbs.append(scheme.abstain_verb)
        return (
            f"You are an answer-key adjudicator for the {scheme.name} task. The\n"
            f"valid labels are: {' | '.join(verbs)}.\n\n"
            "The model produced this raw response:\n"
            f"---\n{raw_text}\n---\n\n"
            "Which label did the model intend? Reply with the literal label word\n"
            "exactly as it appears above. If genuinely unclear, reply 'UNCLEAR'."
        )

    async def judge_tf(self, raw_text: str, scheme, with_unknown: bool) -> str:
        return await self._query(self._build_tf_prompt(raw_text, scheme, with_unknown))


def maybe_build_judge(config: dict) -> Optional[LLMJudge]:
    """Build an LLMJudge from config; return None if disabled or unconfigured.

    Reads `ab_experiment.judge` in the merged config dict. Quiet fallback:
    if the judge can't be built (no key, etc.), prints a one-line warning
    and returns None — runner will skip the fallback and leave UNPARSEABLE
    predictions in place.
    """
    cfg = get_block(config, "main_experiment").get("judge", {}) or {}
    if cfg.get("enabled") is False:
        return None
    try:
        return LLMJudge(
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            model_name=cfg.get("model_name"),
            max_workers=int(cfg.get("max_workers", 5)),
        )
    except Exception as e:
        print(f"[LLMJudge] disabled: {e}")
        return None
