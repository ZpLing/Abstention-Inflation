"""S7 — read the conclusion out of the reasoning trace with an NLI encoder.

For every answerable item this scores two traces with the same probe: the one
the model wrote when the "Unknown" option was absent (S1) and the one it wrote
when the option was present (S2). The premise is the trace with its
"Final answer:" line removed, the hypothesis is the claim being judged.

    entailment    -> the trace concluded True
    contradiction -> the trace concluded False
    neutral       -> the trace reached no conclusion

The trace is read in one of two ways (``--mode``). ``tail`` scores only the
last 1500 characters, which is what fits an NLI encoder in one pass but drops
the body of a long proof. ``windows`` splits the whole trace into overlapping
sentence-packed windows, scores each, and keeps the window carrying the
strongest non-neutral evidence -- the reading that matches the claim actually
being tested, namely that the trace derives the label *somewhere* before the
final token. Taking a max over windows inflates a gold match and a wrong-label
match equally, so the gold-among-decisive rate stays testable against 50%.

Two rates per setting:

    decisive     the trace committed either way (non-neutral)
    gold_aligned the trace committed *and* matched the gold label

Each rate is reported twice: over the full answerable set, which is the
invariance test (if only the final token moved, S1 and S2 agree), and over the
Abstention Inflation subset alone, where S2's rates are the share of
abstentions whose own reasoning had already settled the question.

    python experiments/C3_later_layer_override/S7_reasoning_traces_evaluation/run_S7_reasoning_traces_evaluation.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from loader.dataset_loader import load_judge

sys.path.insert(0, str(Path(__file__).resolve().parent))

# gold answer_idx -> the NLI verdict that agrees with it

#: The NLI encoder and the three verdicts it returns.
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def strip_final_answer(raw: str) -> str:
    """Remove the trailing 'Final answer: X' line so NLI sees CoT only."""
    if not raw:
        return ""
    # cut at the last "Final answer" / "final answer:" if present
    m = re.search(r"\n?\s*final\s+answer\s*:", raw, re.IGNORECASE)
    if m:
        return raw[:m.start()].strip()
    return raw.strip()


def take_tail(text: str, max_chars: int = 1500) -> str:
    """NLI is 512-token limited; keep the last ~1500 chars (~300-400 tok) of CoT."""
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


GOLD_OF_IDX = {0: "entailment", 1: "contradiction"}

CELLS = [
    ("nano",      "gpt-5.4-nano"),
    ("gemini31",  "gemini-3.1-flash-lite"),
    ("dsv4flash", "deepseek-v4-flash"),
]
DATASETS = ("FLD", "FOLIO")
OUT_DIR = ROOT / "results/s7_nli_probe"


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def windows(text, width=1200, overlap=300, cap=12):
    """Pack whole sentences into overlapping windows covering the whole trace."""
    if len(text) <= width:
        return [text]
    sents, out, cur = [s for s in _SENT.split(text) if s.strip()], [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > width:
            out.append(cur)
            cur = (cur[-overlap:] + " " + s) if overlap else s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out[:cap] if len(out) <= cap else (out[:cap - 1] + [out[-1]])


def classify(tok, nli, torch, pairs, batch_size):
    """NLI over (premise, hypothesis) pairs, in order. Returns (label, probs)."""
    device = next(nli.parameters()).device
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk],
                  truncation=True, max_length=512, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(nli(**enc).logits, dim=-1).cpu().tolist()
        for row in probs:
            out.append((LABEL_NAMES[max(range(3), key=lambda k: row[k])], row))
        print(f"    {min(i + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mode", choices=("tail", "windows"), default="windows",
                    help="how much of the trace the probe reads")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print(f"Loading {NLI_MODEL}")
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
    if torch.cuda.is_available():
        nli = nli.cuda()
    elif torch.backends.mps.is_available():
        nli = nli.to("mps")
    print(f"Device: {next(nli.parameters()).device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        by_id = {s.id: s for s in load_judge(ds)}
        for slug, model in CELLS:
            summary = json.loads(
                (ROOT / f"results/tfq/{slug}/ab_summary_{ds}_{model}.json").read_text(encoding="utf-8"))
            items = [s for s in summary["per_sample"]
                     if s["id"] in by_id and by_id[s["id"]].answer_idx in GOLD_OF_IDX]
            n_ai = sum(1 for s in items if s.get("pred_s2") == "UNKNOWN")
            print(f"\n[{ds} / {model}] {len(items)} answerable items "
                  f"({n_ai} of them Abstention Inflation)")

            pairs, owner, traces = [], [], []
            for s in items:
                claim = by_id[s["id"]].question
                for setting in ("s1", "s2"):
                    cot = strip_final_answer(s.get(f"raw_{setting}", "") or "")
                    if not cot:
                        continue
                    parts = ([take_tail(cot)] if args.mode == "tail"
                             else windows(cot))
                    traces.append((s["id"], setting.upper(),
                                   by_id[s["id"]].answer_idx,
                                   s.get("pred_s2") == "UNKNOWN", len(parts)))
                    for part in parts:
                        pairs.append((part, claim))
                        owner.append(len(traces) - 1)
            print(f"  {len(traces)} traces -> {len(pairs)} NLI windows")
            scored = classify(tok, nli, torch, pairs, args.batch_size)

            # per trace: the window carrying the strongest non-neutral evidence
            best = [None] * len(traces)
            for (label, probs), t_i in zip(scored, owner):
                if label == "neutral":
                    continue
                strength = max(probs[0], probs[2])
                if best[t_i] is None or strength > best[t_i][1]:
                    best[t_i] = (label, strength)

            rows = []
            for (sid, setting, idx, is_ai, n_win), b in zip(traces, best):
                verdict = b[0] if b else "neutral"
                rows.append({"id": sid, "setting": setting, "nli": verdict,
                             "n_windows": n_win,
                             "abstention_inflation": is_ai,
                             "decisive": verdict != "neutral",
                             "gold_aligned": verdict == GOLD_OF_IDX[idx]})

            def rates(rs):
                n = len(rs) or 1
                return {"n": len(rs),
                        "decisive_rate": sum(r["decisive"] for r in rs) / n,
                        "gold_aligned_rate": sum(r["gold_aligned"] for r in rs) / n}

            metrics, metrics_ai = {}, {}
            for setting in ("S1", "S2"):
                rs = [r for r in rows if r["setting"] == setting]
                metrics[setting] = rates(rs)
                metrics_ai[setting] = rates([r for r in rs if r["abstention_inflation"]])

            out = OUT_DIR / f"nli_{ds}_{model}.json"
            out.write_text(json.dumps({
                "dataset": ds, "model": model, "nli_model": NLI_MODEL,
                "mode": args.mode,
                "n_answerable": len(items), "n_abstention_inflation": n_ai,
                "metrics": metrics, "metrics_abstention_inflation": metrics_ai,
                "rows": rows,
            }, indent=2), encoding="utf-8")
            for setting in ("S1", "S2"):
                m, a = metrics[setting], metrics_ai[setting]
                print(f"  {setting}: all n={m['n']:4d} decisive {m['decisive_rate']:6.1%} "
                      f"gold-aligned {m['gold_aligned_rate']:6.1%}   |   "
                      f"AI n={a['n']:4d} decisive {a['decisive_rate']:6.1%} "
                      f"gold-aligned {a['gold_aligned_rate']:6.1%}")
            print(f"  saved -> {out.name}")


if __name__ == "__main__":
    main()
