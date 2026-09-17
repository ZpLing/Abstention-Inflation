"""Build the paired S1/S2 accuracy table from the n=500 positional + S1 runs.

Acc(S1) = binary no-Unknown baseline; Acc(S2) = with-Unknown (position C, the
canonical Unknown-last layout). Both are computed over the INTERSECTION of
samples that are valid in *both* conditions (content-filter refusals excluded).
ΔAcc = Acc(S2) − Acc(S1) (signed; negative = accuracy declined when the Unknown
option was added) is a true paired contrast on identical samples.

Writes: <repo root>/positional_accuracy_S1_S2.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results/positional_bias_n500"
MODELS = [
    ("GPT-5.4-nano", "gpt-5.4-nano"),
    ("Gemini-3.1-Flash-Lite", "gemini-2.5-flash-lite"),
    ("DeepSeek-R1", "deepseek-r1-distill-llama-8b"),
]
DATASETS = ["FLD", "FOLIO"]


def load(setting, ds, model):
    f = D / (f"summary_s1_{ds}_{model}.json" if setting == "S1"
             else f"summary_unknown_C_{ds}_{model}.json")  # S2 = Unknown-last
    return json.load(open(f)) if f.exists() else None


def correct(r):
    return (r["pred"] == "A" and r["answer_idx"] == 0) or \
           (r["pred"] == "B" and r["answer_idx"] == 1)


def main():
    lines = [
        "# Accuracy with vs without an Unknown option (unified letter format, n=500)",
        "",
        "**S1** = binary forced choice, `A. True / B. False`. **S2** = the same items "
        "with an `Unknown` option, `A. True / B. False / C. Unknown` (True/False stay "
        "in slots A/B; the abstention option is appended at C). Both use the unified "
        "letter format and the same 500 samples. **Acc** = fraction of the paired "
        "valid items given the correct True/False label; abstentions and unparsed "
        "replies count as incorrect. **ΔAcc = Acc(S2) − Acc(S1)** (negative = "
        "accuracy decline) is computed over the items valid in **both** conditions "
        "(content-filter refusals excluded). Because abstentions are scored wrong, "
        "the S2 decline is driven mainly by the rise in abstention (see the Abs "
        "Rate column), not by degraded True/False reasoning.",
        "",
        "> Treatment: S1→S2 introduces the `Unknown` option, which necessarily also "
        "changes the task instruction (\"true or false\" → \"true, false, or "
        "unknown\") and the option count (2→3). ΔAcc therefore reflects the combined "
        "effect of offering an abstention option, **not** a semantics-isolated "
        "estimate. These are the unified letter-format conditions of this positional "
        "control, **not** the paper's native-TFQ Table 1 S1/S2.",
        "",
        "| Model | Benchmark | Acc(S1) | Acc(S2) | ΔAcc (S2−S1) | Abs Rate(S2) | n(paired) |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    incomplete = []
    for disp, mid in MODELS:
        for ds in DATASETS:
            s1, s2 = load("S1", ds, mid), load("S2", ds, mid)
            if not s1 or not s2:
                lines.append(f"| {disp} | {ds} | — | — | — | — | (missing) |")
                incomplete.append(f"{disp}/{ds}: missing {'S1' if not s1 else ''}{'S2' if not s2 else ''}")
                continue
            if s1.get("complete") is not True or s2.get("complete") is not True:
                incomplete.append(f"{disp}/{ds}: incomplete (S1={s1.get('complete')}, S2={s2.get('complete')})")
            v1 = {r["id"]: r for r in s1["per_sample"] if not r.get("excluded")}
            v2 = {r["id"]: r for r in s2["per_sample"] if not r.get("excluded")}
            ids = sorted(set(v1) & set(v2))
            n = len(ids)
            acc1 = 100 * sum(correct(v1[i]) for i in ids) / n
            acc2 = 100 * sum(correct(v2[i]) for i in ids) / n
            abs_rate_s2 = 100 * sum(1 for i in ids if v2[i]["pred"] == "UNKNOWN") / n
            d = acc2 - acc1  # signed change from adding Unknown; < 0 = accuracy dropped
            sign = "−" if d < 0 else "+"
            lines.append(
                f"| {disp} | {ds} | {acc1:.1f}% | {acc2:.1f}% | "
                f"{sign}{abs(d):.1f}pp | {abs_rate_s2:.1f}% | {n} |"
            )
    lines += [
        "",
        "*Endpoints actually queried: GPT-5.4-nano→`gpt-5.4-nano`, "
        "Gemini-3.1-Flash-Lite→`gemini-2.5-flash-lite`, "
        "DeepSeek-R1→`deepseek-r1-distill-llama-8b`.*",
    ]
    if incomplete:
        lines += ["", "## Incomplete / missing (excluded)", ""] + [f"- {c}" for c in incomplete]
    out = ROOT / "positional_accuracy_S1_S2.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
