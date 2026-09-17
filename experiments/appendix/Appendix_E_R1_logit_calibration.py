"""App. E / R1 — option-presence logit calibration, run on open weights.

R1 asks whether the abstention that the ``Unknown'' option installs can be
removed at inference time by subtracting the structural prior it adds to the
abstention token. The gateway used for the API models returns no logprobs, so
this runs on OLMo-3-7B-Instruct, the same checkpoint S8 probes.

The decision logprobs are read with a forward pass rather than by locating
``Final answer:`` inside a sampled string. The model first writes its reasoning;
the reasoning is then truncated at ``Final answer:`` (or the whole generation is
kept when it never got there), ``Final answer:`` is appended, and one forward
pass gives the exact next-token distribution at the position where the verb is
chosen. Nothing is lost to a parse failure, which the earlier API smoke run lost
roughly half its items to.

    beta_hat   the abstain token's mean *margin* over the better of the two
               concrete labels, measured on the S2 calibration split:
               mean( lp(Unknown) - max(lp(True), lp(False)) )
    R1         argmax over {lp(True), lp(False), lp(Unknown) - beta_hat}
               on the held-out split

    Estimating beta_hat as the mean lp(Unknown) alone -- the first reading of
    Theorem 1's phi_struct -- does not work: the logprobs are all negative, so
    subtracting their mean re-centres Unknown near zero while the concrete
    labels stay where they are, and the correction pushes *towards* abstention.
    On OLMo-3-7B-Instruct it raised Abs Rate from 28.4% to 40.0% on FOLIO and
    from 70.0% to 82.0% on FLD. The margin form subtracts the advantage the
    option's presence actually confers, which is what the theorem describes.

    python experiments/appendix/Appendix_E_R1_logit_calibration.py \
        --model_path <olmo-3-7b-instruct>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from loader.dataset_loader import load_judge
from infra.label_scheme import get_scheme
from infra.prompts import build_judge_s1_prompt, build_judge_s2_prompt

DATASETS = ("FLD", "FOLIO")
_FINAL = re.compile(r"final\s*answer\s*:", re.IGNORECASE)


def first_token_id(tok, word):
    """Token id the model would emit first for `word` after ``Final answer:``."""
    for form in (f" {word}", word):
        ids = tok(form, add_special_tokens=False).input_ids
        if ids:
            return ids[0]
    raise ValueError(word)


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size):
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j][enc.input_ids.shape[1]:],
                                  skip_special_tokens=True))
        print(f"    gen {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


@torch.no_grad()
def decision_logprobs(model, tok, texts, cand_ids, batch_size):
    """Next-token logprob of each candidate at the final-answer position."""
    rows = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=3584).to(model.device)
        logits = model(**enc).logits
        # left padding -> the last column is the next-token position for all rows
        lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
        for j in range(len(chunk)):
            rows.append({k: lp[j, v].item() for k, v in cand_ids.items()})
        print(f"    score {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return rows


def cut_at_final(text):
    m = _FINAL.search(text)
    return text[:m.start()].rstrip() if m else text.rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", default="olmo3-instruct")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out_dir", default=str(ROOT / "results/remedy_r1"))
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"loaded {args.model_path} on {model.device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview = []

    for ds in args.datasets:
        scheme = get_scheme(ds)
        samples = [s for s in load_judge(ds) if s.answer_idx in (0, 1)]
        cand = {"A": first_token_id(tok, scheme.pos_verb),
                "B": first_token_id(tok, scheme.neg_verb),
                "UNKNOWN": first_token_id(tok, scheme.abstain_verb)}
        print(f"\n[{ds}] {len(samples)} items; candidate ids {cand}", flush=True)

        preds, lps = {}, {}
        for setting, build in (("S1", build_judge_s1_prompt),
                               ("S2", build_judge_s2_prompt)):
            prompts = [render(tok, build(scheme, s.question, s.context)) for s in samples]
            gens = generate(model, tok, prompts, args.max_new_tokens, args.batch_size)
            scored = [p + cut_at_final(g) + "\nFinal answer:"
                      for p, g in zip(prompts, gens)]
            rows = decision_logprobs(model, tok, scored, cand, args.batch_size)
            keys = ("A", "B") if setting == "S1" else ("A", "B", "UNKNOWN")
            preds[setting] = [max(keys, key=lambda k: r[k]) for r in rows]
            lps[setting] = rows

        # 50/50 calibration / test split
        import random
        idx = list(range(len(samples)))
        random.Random(args.seed).shuffle(idx)
        calib, test = idx[:len(idx) // 2], idx[len(idx) // 2:]
        margins = [lps["S2"][i]["UNKNOWN"] - max(lps["S2"][i]["A"], lps["S2"][i]["B"])
                   for i in calib]
        beta_hat = sum(margins) / len(margins)

        def r1_pred(i):
            r = dict(lps["S2"][i])
            r["UNKNOWN"] -= beta_hat
            return max(("A", "B", "UNKNOWN"), key=lambda k: r[k])

        def stats(pred_of, on):
            gold = [chr(ord("A") + samples[i].answer_idx) for i in on]
            p = [pred_of(i) for i in on]
            return {"n": len(on),
                    "acc": sum(a == b for a, b in zip(p, gold)) / len(on),
                    "abs_rate": sum(x == "UNKNOWN" for x in p) / len(on)}

        s1 = stats(lambda i: preds["S1"][i], test)
        s2 = stats(lambda i: preds["S2"][i], test)
        r1 = stats(r1_pred, test)
        row = {"model": args.model_name, "dataset": ds, "n_items": len(samples),
               "beta_hat": beta_hat, "S1_CoT": s1, "S2_CoT": s2,
               "R1": {**r1, "delta_acc_vs_s2": r1["acc"] - s2["acc"],
                      "delta_abs_vs_s2": r1["abs_rate"] - s2["abs_rate"]}}
        overview.append(row)
        (out_dir / f"r1_{ds}_{args.model_name}.json").write_text(json.dumps({
            **row, "per_sample": [
                {"id": samples[i].id, "answer_idx": samples[i].answer_idx,
                 "split": "test" if i in set(test) else "calib",
                 "pred_s1": preds["S1"][i], "pred_s2": preds["S2"][i],
                 "pred_r1": r1_pred(i), "lp_s2": lps["S2"][i]}
                for i in range(len(samples))]}, indent=2))
        print(f"  beta_hat={beta_hat:.3f}")
        for name, v in (("S1", s1), ("S2", s2), ("R1", r1)):
            print(f"  {name}: acc={v['acc']:.1%} abs={v['abs_rate']:.1%} (n={v['n']})")

    rows = []
    for f in sorted(out_dir.glob(f"r1_*_{args.model_name}.json")):
        d = json.loads(f.read_text())
        rows.append({k: v for k, v in d.items() if k != "per_sample"})
    (out_dir / "r1_overview.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
