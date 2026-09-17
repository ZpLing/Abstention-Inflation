"""Local HF model wrapper + answer-token hidden-state extraction.

The probe experiment (§四 小实验 1D) needs per-layer residual stream activations
at the answer-token position, which closed-API models do not expose. This module
loads a local causal-LM via transformers and provides:

    - load_probe_model(path, device, dtype) → (model, tokenizer)
    - extract_answer_token_hidden_states(model, tokenizer, prompts, ...)
        → np.ndarray  shape: (n_prompts, n_layers, hidden_dim)
    - generate_with_ablation(model, tokenizer, prompt, layer, direction, ...)
        → str (decoded continuation)

Answer-token position (PLAN_v2 §四 小实验 1D): the *last input token* —
that is the position from which the model will sample its first output token
and which carries the most direct readout of "what letter would I emit". This
matches the convention used in Arditi+ 2024 and shrugger.

Memory: hidden states are huge; we move each (n_layers, hidden_dim) slice to
CPU + numpy immediately after the forward pass. fp16 keeps the batch
manageable on Apple-silicon `mps`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# torch + transformers are heavy; import lazily inside helpers so importing
# the package on a host without ML libs (e.g. for IDE inspection of the
# top-level orchestrator) does not blow up.


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_probe_model(model_path: str, *,
                      device: Optional[str] = None,
                      dtype: str = "float16"):
    """Load a local causal-LM + tokenizer for probing.

    `model_path` is a local directory (e.g. "models/gemma-4-E2B-it"). We avoid
    online resolution because the probe runs offline by design.

    Returns (model, tokenizer, device). The model is in eval mode with
    `output_hidden_states=True` set on its config, so subsequent forward
    passes return per-layer hidden states without per-call kwarg plumbing.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"[probe_1d] local model not found at {path}. "
            "Set probe_1d.model_path in config or download the model first."
        )
    dev = _resolve_device(device)
    torch_dtype = {"float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32}[dtype]

    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.output_hidden_states = True
    model.eval()
    model.to(dev)
    return model, tok, dev


def _format_prompt(messages: List[dict], tokenizer) -> str:
    """Render a messages list into a single string for tokenization.

    Prefers the model's chat template (which exists for gemma / qwen / llama
    instruct checkpoints) and falls back to a flat user-only concatenation
    so base models without a chat template still work.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n\n".join(m["content"] for m in messages)


def extract_answer_token_hidden_states(
    model, tokenizer, device: str,
    prompts: List[List[dict]],
    *, batch_size: int = 4,
    max_length: int = 2048,
) -> np.ndarray:
    """Forward each prompt; extract hidden_states[ℓ][:, -1, :] for every layer.

    Returns shape (n_prompts, n_layers, hidden_dim). Layer 0 is the embedding
    output and the last entry is the post-final-block residual stream — this
    matches the indexing convention every per-layer probe in the literature
    uses, so the runner can iterate `range(n_layers)` without remapping.
    """
    import torch
    n = len(prompts)
    if n == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)

    rendered = [_format_prompt(p, tokenizer) for p in prompts]
    out_chunks: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        batch = rendered[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        # out.hidden_states is a tuple of (n_layers+1) tensors,
        # each shape (batch, seq, hidden).
        hs = torch.stack(out.hidden_states, dim=0)  # (L, B, T, H)
        # Pick the last *non-padding* token position per row; with right-padding
        # that is `attention_mask.sum(-1) - 1`.
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=-1) - 1  # (B,)
        # gather along T dimension per (L, B); use advanced indexing.
        L, B, T, H = hs.shape
        gathered = hs[:, torch.arange(B), last_idx, :]  # (L, B, H)
        gathered = gathered.transpose(0, 1).contiguous()  # (B, L, H)
        out_chunks.append(gathered.float().cpu().numpy())
        del out, hs, gathered
    return np.concatenate(out_chunks, axis=0)  # (n_prompts, L, H)


def generate_with_ablation(
    model, tokenizer, device: str,
    prompt_messages: List[dict],
    *, layer: int, direction: np.ndarray,
    max_new_tokens: int = 200,
    apply_to_all_positions: bool = True,
) -> str:
    """Generate a continuation with `direction` projected out of layer `layer`.

    The hook subtracts (h · ê)ê from the layer's residual stream output where
    ê is the unit-normalized `direction`. `apply_to_all_positions=True` ablates
    every token; this is the standard Arditi et al. setup for causal validation
    of a refusal-style direction. Set False to ablate only the answer-token
    position (rarely useful — the model may emit a Unknown token before that
    position even sees the ablation).
    """
    import torch
    text = _format_prompt(prompt_messages, tokenizer)
    enc = tokenizer(text, return_tensors="pt").to(device)

    e = torch.tensor(direction, dtype=next(model.parameters()).dtype, device=device)
    e = e / (e.norm() + 1e-8)

    target_module = _resolve_layer_module(model, layer)

    def _hook(_module, _input, output):
        # output may be a tensor or a tuple (residual, ...). Handle both.
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        if apply_to_all_positions:
            proj = (h @ e).unsqueeze(-1) * e
            h = h - proj
        else:
            # only the last position
            last = h[..., -1:, :]
            proj = (last @ e).unsqueeze(-1) * e
            h = torch.cat([h[..., :-1, :], last - proj], dim=-2)
        if rest is not None:
            return (h,) + rest
        return h

    handle = target_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0, enc["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        handle.remove()


def _resolve_layer_module(model, layer: int):
    """Return the nn.Module whose forward output is the post-block residual at
    `layer` (0-indexed transformer block, NOT including the embedding layer).

    Handles common HuggingFace causal-LM layouts. Negative `layer` indexes
    from the end (-1 = last block).
    """
    blocks = _find_transformer_blocks(model)
    if not blocks:
        raise RuntimeError(
            "[probe_1d] could not locate transformer blocks for ablation hook; "
            f"model class={type(model).__name__}"
        )
    if layer < 0:
        layer = len(blocks) + layer
    if layer < 0 or layer >= len(blocks):
        raise IndexError(
            f"[probe_1d] layer {layer} out of range (n_blocks={len(blocks)})"
        )
    return blocks[layer]


def _find_transformer_blocks(model):
    """Best-effort search for the ModuleList of transformer blocks."""
    candidates = ["layers", "h", "blocks", "decoder.layers"]
    bases = [model, getattr(model, "model", None), getattr(model, "transformer", None)]
    for base in bases:
        if base is None:
            continue
        for name in candidates:
            cur = base
            ok = True
            for part in name.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__"):
                return cur
    return None


def n_layers_from_hidden_states(hs: np.ndarray) -> int:
    """`hs` is (n_prompts, L, H). Returns L."""
    return int(hs.shape[1])


def n_blocks_from_model(model) -> int:
    blocks = _find_transformer_blocks(model)
    return len(blocks) if blocks is not None else 0
