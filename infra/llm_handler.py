import asyncio
from typing import Any, Dict, List

import openai
from tqdm.asyncio import tqdm_asyncio  # Progress bar for async tasks


class LLMHandler:
    """
    Dedicated diplomat responsible for interacting with all large language model APIs.
    It encapsulates client initialization, concurrency control, API calls, and error handling.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration and set up API client and concurrency controller.
        """
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.model_name = config.get("model_name")

        if not self.api_key:
            raise ValueError("API key not set in configuration file or secrets.yaml.")

        # Initialize async OpenAI client
        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # Get concurrency count from configuration and create semaphore
        max_workers = self.config.get("max_workers", 5)
        self.semaphore = asyncio.Semaphore(max_workers)
        self.max_tokens = self.config.get("max_tokens", 4096)
        # Provider-specific JSON body extensions (e.g. DashScope requires
        # `enable_thinking: false` for non-streaming Qwen3 calls).
        self.extra_body = self.config.get("extra_body") or {}

    async def _query_single(self, message: List[Dict[str, str]]) -> str:
        """
        Make API call for a single message, including error handling and concurrency control.

        For reasoning models (e.g. deepseek-r1, o1), the OpenAI-compatible API
        returns the chain-of-thought separately as `message.reasoning_content`,
        not embedded in `message.content`. We merge both back into one string
        formatted as `Reasoning: ...\\n\\n<final answer>` so the existing
        Evaluator.extract_reasoning / extract_final_answer_line pipeline picks
        up both the reasoning (read by the S7 probe) and the answer (Acc / F1).
        """
        data = {
            "model": self.model_name,
            "messages": message,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

        call_kwargs = dict(data)
        if self.extra_body:
            call_kwargs["extra_body"] = self.extra_body

        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(**call_kwargs)
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
                if reasoning:
                    return f"Reasoning: {reasoning}\n\n{content}"
                return content
            except Exception as e:
                print(f"Error occurred during API call: {e}")
                print(f"Request data that caused error: {call_kwargs}")
                return "__API_ERROR__"  # Return a clear error identifier

    async def batch_query(self, messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Receive a list of messages, execute all API calls concurrently, and return results in original order.
        This is the core external method that encapsulates the entire concurrent execution loop.

        Args:
            messages: A list of messages, where each element is a message list conforming to OpenAI format.

        Returns:
            A list of strings containing all API responses in corresponding order.
        """
        if not messages:
            return []

        print(f"Starting batch concurrent query for {len(messages)} messages...")

        # Create all async tasks that need to be executed
        tasks = [self._query_single(msg) for msg in messages]

        # Use tqdm_asyncio.gather to execute tasks and display progress bar
        results = await tqdm_asyncio.gather(*tasks, desc=f"Querying {self.model_name}")

        print("All queries completed.")
        return results

    async def _query_single_with_logprobs(
        self,
        message: List[Dict[str, str]],
        top_logprobs: int = 20,
        max_tokens: int = 4,
        return_all_positions: bool = False,
    ) -> Dict[str, Any]:
        """Single call returning {content, top_logprobs_first_token}, plus,
        if return_all_positions, also {tokens: [{token, logprob, top_logprobs}]}.

        On API error returns {"content": "__API_ERROR__", ...empty fields}.
        """
        data = {
            "model": self.model_name,
            "messages": message,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        call_kwargs = dict(data)
        if self.extra_body:
            call_kwargs["extra_body"] = self.extra_body

        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(**call_kwargs)
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                top_first = []
                tokens = []
                lp = getattr(choice, "logprobs", None)
                if lp is not None and getattr(lp, "content", None):
                    seq = lp.content
                    if seq:
                        for entry in getattr(seq[0], "top_logprobs", []) or []:
                            top_first.append(
                                {"token": entry.token, "logprob": entry.logprob}
                            )
                    if return_all_positions:
                        for pos in seq:
                            tlp = []
                            for entry in getattr(pos, "top_logprobs", []) or []:
                                tlp.append(
                                    {"token": entry.token, "logprob": entry.logprob}
                                )
                            tokens.append(
                                {
                                    "token": pos.token,
                                    "logprob": pos.logprob,
                                    "top_logprobs": tlp,
                                }
                            )
                return {
                    "content": content,
                    "top_logprobs_first_token": top_first,
                    "tokens": tokens,
                }
            except Exception as e:
                print(f"Logprob query error: {e}")
                return {
                    "content": "__API_ERROR__",
                    "top_logprobs_first_token": [],
                    "tokens": [],
                }

    async def batch_query_with_logprobs(
        self,
        messages: List[List[Dict[str, str]]],
        top_logprobs: int = 20,
        max_tokens: int = 4,
        return_all_positions: bool = False,
    ) -> List[Dict[str, Any]]:
        if not messages:
            return []
        tasks = [
            self._query_single_with_logprobs(
                m,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
                return_all_positions=return_all_positions,
            )
            for m in messages
        ]
        return await tqdm_asyncio.gather(
            *tasks, desc=f"Logprob query {self.model_name}"
        )
