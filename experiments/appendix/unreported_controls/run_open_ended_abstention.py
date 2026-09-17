#!/usr/bin/env python3
"""Open-ended Abstention Inflation test (reviewer point #5).

Tests whether an abstention *affordance* expressed in natural language (not a
discrete option slot) inflates abstention on questions the model can answer, in
a free-form generation setting. Uses the SelfAware benchmark (Yin et al., ACL
Findings 2023), which pairs answerable questions (SQuAD/HotpotQA/TriviaQA) with
genuinely unanswerable ones.

Conditions (the "add an escape affordance" manipulation, in prose):
  O1  no license          — "Answer the following question."
  O2  prose license       — "... if it cannot be determined, it's acceptable to say so."
  O3  nonce-word license   — "... if you cannot determine it, reply with 'Cerulean'."
                            (open-ended analog of the paper's random-word control)

Generator: gpt-5.4-nano.  Abstention/correctness judge: gemini-2.5-flash-lite.
Both via the gateway OpenAI-compatible gateway (creds from ./config).

Outputs results/open_ended/selfaware_gpt5nano.json with per-sample records and a
summary block. Re-runs reuse cached generations+judgments (keyed by qid+cond).
"""
import os, sys, json, asyncio, random, argparse
import yaml, openai

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "open_ended")
OUT_PATH = os.path.join(OUT_DIR, "selfaware_gpt5nano.json")
DATA = os.path.join(REPO, "data", "open_ended", "SelfAware.json")

GEN_MODEL = "gpt-5.4-nano"
JUDGE_MODEL = "gemini-2.5-flash-lite"

# each condition = {user, system?}. O1=no license, O2=prose license, O3=nonce-word
# license (open-ended analog of the paper's random-word control), O4=deployment
# system-prompt caution (most ecologically valid "natural setting").
CONDITIONS = {
    "O1": {"user": "Answer the following question.\n\nQuestion: {q}"},
    "O2": {"user": ("Answer the following question. If the answer cannot be determined "
                    "or is genuinely unknown, it is acceptable to say so rather than guess."
                    "\n\nQuestion: {q}")},
    "O3": {"user": ("Answer the following question. If you cannot determine the answer, "
                    "reply with the single word \"Cerulean\"."
                    "\n\nQuestion: {q}")},
    "O4": {"system": ("You are a careful assistant. If you are not confident that you know "
                      "the correct answer, it is better to say you don't know than to guess."),
           "user": "Answer the following question.\n\nQuestion: {q}"},
}

JUDGE_TMPL = """You are grading a model's response to a question.

Question: {q}
{refblock}Model response:
\"\"\"{resp}\"\"\"

Decide two things:
1. verdict — did the model ABSTAIN or ANSWER?
   - "ABSTAIN": it declined; said it does not know / cannot be determined / is \
unknown / uncertain; refused; or replied only with the placeholder word "Cerulean".
   - "ANSWER": it committed to a substantive answer.
2. correct — only if verdict is ANSWER and reference answers are given: true if the \
committed answer matches ANY reference (allow paraphrase/synonyms/extra words), \
false if it committed to a wrong answer. Use null if it abstained or no reference.

Respond with STRICT JSON on one line, nothing else:
{{"verdict": "ANSWER" or "ABSTAIN", "correct": true or false or null}}"""


def load_client():
    cfg = yaml.safe_load(open(os.path.join(REPO, "config")))["llm"]
    return openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def subsample(n_each, seed=42):
    d = json.load(open(DATA))["example"]
    # answerable pool = TriviaQA subset only: self-contained factual questions the
    # model can answer from parametric knowledge (SQuAD/HotpotQA items are
    # context-dependent and unfair as a "model-knows" pool).
    ans = [x for x in d if x.get("answerable") is True and "trivia" in x["source"].lower()]
    una = [x for x in d if x.get("answerable") is False]
    rng = random.Random(seed)
    rng.shuffle(ans); rng.shuffle(una)
    items = []
    for x in ans[:n_each]:
        items.append({"qid": x["question_id"], "q": x["question"],
                      "pool": "answerable", "ref": x.get("answer") or []})
    for x in una[:n_each]:
        items.append({"qid": x["question_id"], "q": x["question"],
                      "pool": "unanswerable", "ref": []})
    return items


async def gen_one(client, sem, item, cond):
    spec = CONDITIONS[cond]
    messages = []
    if spec.get("system"):
        messages.append({"role": "system", "content": spec["system"]})
    messages.append({"role": "user", "content": spec["user"].format(q=item["q"])})
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=messages,
                    temperature=0.0, max_tokens=2048)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 2:
                    return f"__API_ERROR__ {type(e).__name__}"
                await asyncio.sleep(2 * (attempt + 1))


async def judge_one(client, sem, item, resp):
    if resp.startswith("__API_ERROR__"):
        return {"verdict": "ERROR", "correct": None}
    refblock = ""
    if item["pool"] == "answerable" and item["ref"]:
        refblock = "Reference answer(s): " + " | ".join(map(str, item["ref"])) + "\n\n"
    prompt = JUDGE_TMPL.format(q=item["q"], refblock=refblock, resp=resp[:4000])
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=2000)
                txt = (r.choices[0].message.content or "").strip()
                s, e = txt.find("{"), txt.rfind("}")
                obj = json.loads(txt[s:e + 1])
                v = obj.get("verdict", "").upper()
                if v not in ("ANSWER", "ABSTAIN"):
                    v = "ABSTAIN" if v.startswith("ABST") else "ANSWER"
                return {"verdict": v, "correct": obj.get("correct")}
            except Exception as e:
                if attempt == 2:
                    return {"verdict": "PARSE_ERROR", "correct": None}
                await asyncio.sleep(2 * (attempt + 1))


async def main(n_each, max_workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    client = load_client()
    sem = asyncio.Semaphore(max_workers)
    items = subsample(n_each)
    print(f"{len(items)} items ({n_each}/pool) x {len(CONDITIONS)} conditions", flush=True)

    # cache
    cache = {}
    if os.path.exists(OUT_PATH):
        try:
            for r in json.load(open(OUT_PATH)).get("per_sample", []):
                cache[(r["qid"], r["cond"])] = r
        except Exception:
            pass

    todo = [(it, c) for it in items for c in CONDITIONS if (it["qid"], c) not in cache]
    print(f"cached={len(cache)}  to_run={len(todo)}", flush=True)

    # generate
    gens = await asyncio.gather(*[gen_one(client, sem, it, c) for it, c in todo])
    # judge
    judgments = await asyncio.gather(*[judge_one(client, sem, it, g)
                                       for (it, c), g in zip(todo, gens)])
    for (it, c), g, j in zip(todo, gens, judgments):
        # unanswerable items have no reference answer -> correctness is undefined
        correct = None if it["pool"] == "unanswerable" else j["correct"]
        cache[(it["qid"], c)] = {"qid": it["qid"], "cond": c, "pool": it["pool"],
                                 "q": it["q"], "ref": it["ref"], "response": g,
                                 "verdict": j["verdict"], "correct": correct}

    per_sample = list(cache.values())
    summary = compute_summary(per_sample)
    json.dump({"generator": GEN_MODEL, "judge": JUDGE_MODEL, "n_each": n_each,
               "summary": summary, "per_sample": per_sample},
              open(OUT_PATH, "w"), indent=2, ensure_ascii=False)
    print_summary(summary)
    print(f"\nsaved -> {OUT_PATH}")


def compute_summary(ps):
    def sel(pool, cond):
        return [r for r in ps if r["pool"] == pool and r["cond"] == cond
                and r["verdict"] in ("ANSWER", "ABSTAIN")]
    out = {}
    for pool in ("answerable", "unanswerable"):
        out[pool] = {}
        for cond in CONDITIONS:
            rows = sel(pool, cond)
            n = len(rows)
            ab = sum(r["verdict"] == "ABSTAIN" for r in rows)
            ans = [r for r in rows if r["verdict"] == "ANSWER"]
            acc = (sum(r["correct"] is True for r in ans) / len(ans)) if ans else None
            out[pool][cond] = {"n": n, "abstain_rate": round(ab / n, 3) if n else None,
                               "acc_on_answered": round(acc, 3) if acc is not None else None}
    # bad-abstention decomposition on answerable: among items ANSWERED CORRECTLY in O1,
    # how many abstained in O2 / O3?
    o1correct = {r["qid"] for r in ps if r["pool"] == "answerable" and r["cond"] == "O1"
                 and r["verdict"] == "ANSWER" and r["correct"] is True}
    out["inflation_answerable"] = {"n_correct_in_O1": len(o1correct)}
    for cond in ("O2", "O3"):
        byqid = {r["qid"]: r for r in ps if r["pool"] == "answerable" and r["cond"] == cond}
        bad = sum(1 for q in o1correct if byqid.get(q, {}).get("verdict") == "ABSTAIN")
        out["inflation_answerable"][cond] = {
            "bad_abstain_on_known": bad,
            "bad_abstain_rate": round(bad / len(o1correct), 3) if o1correct else None}
    return out


def print_summary(s):
    print("\n" + "=" * 66)
    print("OPEN-ENDED ABSTENTION INFLATION  (SelfAware, gpt-5.4-nano)")
    print("=" * 66)
    for pool in ("answerable", "unanswerable"):
        print(f"\n[{pool}]  abstain_rate by condition (acc among answered):")
        for cond in CONDITIONS:
            c = s[pool][cond]
            acc = f"  acc={c['acc_on_answered']}" if c["acc_on_answered"] is not None else ""
            print(f"   {cond}: abstain={c['abstain_rate']}  (n={c['n']}){acc}")
    ia = s["inflation_answerable"]
    print(f"\n[INFLATION]  items answered CORRECTLY under O1 (model demonstrably knows): "
          f"{ia['n_correct_in_O1']}")
    for cond in ("O2", "O3"):
        print(f"   under {cond}: {ia[cond]['bad_abstain_on_known']} of those abstained "
              f"= {ia[cond]['bad_abstain_rate']} 'bad' abstention rate")
    a, u = s["answerable"], s["unanswerable"]
    for cond in ("O2", "O3"):
        da = a[cond]["abstain_rate"] - a["O1"]["abstain_rate"]
        du = u[cond]["abstain_rate"] - u["O1"]["abstain_rate"]
        print(f"\n   {cond} vs O1:  Δabstain answerable = {da:+.3f}   "
              f"Δabstain unanswerable = {du:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_each", type=int, default=150)
    ap.add_argument("--max_workers", type=int, default=24)
    args = ap.parse_args()
    asyncio.run(main(args.n_each, args.max_workers))
