"""Derive the S10 Factor-Analysis figure data from result files.

Replaces tables that were hardcoded inside the plotting script. Three errors
were only visible once the numbers came from the runs again:

  * the S10(a) T=0 column had been taken from a different sample set than the
    sweep it is plotted against;
  * the S10(b) bins ran to "16+" although ``dataset/FLD.json`` only carries
    proof depths 1-8;
  * every FLD figure silently depended on ``infra/label_scheme.py``, whose FLD
    verbs changed from Proved/Disproved to True/False *after* the runs, so the
    stored outputs re-parsed to the wrong label until the FLD scheme was taught
    to accept both surface forms.

Sources are named explicitly below and never globbed. Several result
directories hold the *same sample ids under different experimental
conditions* -- ``ab_e_option_baseline`` is an E-option variant, ``ab_s5`` is an
S5 rerun -- so a glob plus first-file-wins silently mixes conditions. The FLD
pairing here is the one the repo already treats as canonical, in
``run_S10_local_Gemma_inference.load_paired_sample_ids``.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from infra.evaluator import Evaluator          # noqa: E402
from infra.label_scheme import get_scheme      # noqa: E402
from infra.result_schema import load_summary   # noqa: E402

EV = Evaluator()
RESULTS = ROOT / "results"
SWEEP_DIR = RESULTS / "temperature_sweep"
#: Second pass covering the 300 items per dataset the first sweep did not draw.
#: The paper reports n=500; the first sweep drew the pooled batch-1 + batch-2
#: 200, so the rest is pooled in here rather than re-running those 200.
SWEEP_P2_DIR = RESULTS / "temperature_sweep_p2"
#: Sweeps run on locally served checkpoints, where the temperature setting is
#: genuinely applied. Different file naming and schema from the API sweep
#: above -- `ab_summary_<ds>_<tag>_T<temp>.json` with `raw_s2` per item rather
#: than `summary_T<temp>_<ds>_<tag>.json` with `raw` -- so they are read by
#: :func:`_local_temperature_table` and merged in.
#: display name -> (directory, model tag as it appears in the file names).
#: Everything written in the S10 runner's schema is read here, which since the
#: Gemini re-run includes an API model as well as the locally served ones: the
#: point of the entry is the schema, not where the tokens came from.
#:
#: The Gemini cells come from `s10_temp_gemini31`. The older sweep under
#: `temperature_sweep{,_p2}` was collected on a model the paper no longer
#: tests and has been retired, so it is superseded rather than pooled.
#:
#: OLMo is the top_k=20 sweep. Its untruncated twin decoded into token soup
#: above T=1 -- 13-20% of words at T>=1.5 came from the vocabulary the model
#: uses at T=0, and no item at T=2.0 emitted a `Final answer:` line -- so those
#: cells measure decoding collapse, not abstention.
LOCAL_TEMP_DIRS = {
    # The key must match MODELS/MODEL_NAMES *exactly*, newline included, or the
    # figure treats Gemini as an unknown extra and hands it a fallback colour --
    # giving one model two different colours across the two panels.
    "Gemini-3.1-\nFlash-Lite": (RESULTS / "s10_temp_gemini31", "gemini-3.1-flash-lite"),
    "Olmo-3-7B-Instruct":      (RESULTS / "s10_temp_olmo_topk20", "olmo3-instruct"),
    # Qwen is not on the curve. Both attempts fail the same way and for the same
    # reason: Qwen3.5 reasons at length before committing, so on FLD the median
    # generation *is* the cap -- 8192 tokens exactly at 4B, with 76-96% of items
    # truncated and trusted share 5-30%. The 9B run at 24576 was no better
    # (36-38%). Only its FOLIO cells at T<=1.0 parse (78-81%), and half a model
    # is not a curve. Measured in tokens, not characters: the generations carry
    # enough LaTeX that a character-length check reads far below the cap and
    # hides the truncation entirely.
}

TEMPS = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
_TEMP_TAGS = {0.3: "T0p3", 0.7: "T0p7", 1.0: "T1p0", 1.5: "T1p5", 2.0: "T2p0"}

#: paper display name -> the ``model`` field written into the result files.
#: S10(b) difficulty uses all three; S10(a) temperature uses only the models
#: whose temperature setting the endpoint actually applies -- see
#: :data:`TEMPERATURE_HONOURED`.
MODELS = {
    "DeepSeek-V4-Flash":             "deepseek-v4-flash",
    "GPT-5.4-nano":            "gpt-5.4-nano",
    "Gemini-3.1-\nFlash-Lite": "gemini-3.1-flash-lite",
}

#: Models the gateway actually varies temperature for. Measured, not assumed:
#: repeat one prompt N times and count distinct outputs. At T=0 greedy decoding
#: must be deterministic, so a model that returns N distinct outputs there is
#: not being given the setting at all.
#:
#:     DeepSeek-V4-Flash    T=0: 24/24 distinct   T=1: 24/24   T=2: 24/24
#:     GPT-5.4-nano   T=0: 12/12 distinct                T=2: 12/12
#:     Gemini         T=0:  5/24 distinct   T=1: 24/24   T=2: 24/24
#:
#: The first two behave identically at every setting, so their flat
#: Abs-Rate-vs-T curves are flat because nothing varied. Including them would
#: present "no effect of temperature" as a finding when temperature was never
#: applied. Their raw files are kept; they are simply not temperature evidence.
#: API models whose endpoint actually applies `temperature`, and whose sweep is
#: still read through the API code path below. Empty since the Gemini re-run:
#: 3.1 is read from :data:`LOCAL_TEMP_DIRS` instead, in the same schema as the
#: served checkpoints, so all curves share one parser and one item set.
#: Probed rather than assumed -- 30-60 draws per temperature, comparing the
#: answer distribution: gemini-3.1-flash-lite flattens with temperature, while
#: gpt-5.4-nano, deepseek-r1 and kimi-k3 return the same distribution at T=0
#: and T=2 and are therefore not plotted at all.
TEMPERATURE_HONOURED: set[str] = set()

#: (model tag, dataset) -> the two batch files whose union is the 200-item set
#: the temperature sweep drew from. Used for the T=0 baseline only.
BASELINE_SOURCES = {
    ("deepseek-v4-flash", "FLD"): [
        "ab_e_option_baseline/ab_summary_FLD_deepseek-v4-flash.json",
        "ab_deepseek_batch2/ab_summary_FLD_deepseek-v4-flash.json"],
    ("deepseek-v4-flash", "FOLIO"): [
        "ab_followup/ab_summary_FOLIO_deepseek-v4-flash.json",
        "ab_deepseek_batch2/ab_summary_FOLIO_deepseek-v4-flash.json"],
    ("gpt-5.4-nano", "FLD"): [
        "ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json",
        "ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json"],
    ("gpt-5.4-nano", "FOLIO"): [
        "ab_gpt5_nano/ab_summary_FOLIO_gpt-5.4-nano.json",
        "ab_nano_batch2/ab_summary_FOLIO_gpt-5.4-nano.json"],
    ("gemini-3.1-flash-lite", "FLD"): [
        "ab_gemini_flash_lite/ab_summary_FLD_gemini-3.1-flash-lite.json",
        "ab_gemini_batch2/ab_summary_FLD_gemini-3.1-flash-lite.json"],
    ("gemini-3.1-flash-lite", "FOLIO"): [
        "ab_gemini_flash_lite/ab_summary_FOLIO_gemini-3.1-flash-lite.json",
        "ab_gemini_batch2/ab_summary_FOLIO_gemini-3.1-flash-lite.json"],
}

#: S10(b) reads the paper's FLD n=500 main run and nothing else -- pooling the
#: n=100 batches in as well would mix conditions for no extra coverage.
#: S10(b) stratifies the S2 cell by proof depth, so it reads the same S2 run
#: the rest of the paper reports: the abstain-verb-last cell of S11, which is
#: `build_judge_s2_prompt` byte for byte.
DIFFICULTY_SOURCE = {
    "deepseek-v4-flash":
        "positional_bias/summary_unknown_C_FLD_deepseek-v4-flash.json",
    "gpt-5.4-nano":
        "positional_bias/summary_unknown_C_FLD_gpt-5.4-nano.json",
    "gemini-3.1-flash-lite":
        "positional_bias/summary_unknown_C_FLD_gemini-3.1-flash-lite.json",
}


def _abs_rate(preds) -> float:
    return 100.0 * sum(1 for p in preds if p == "UNKNOWN") / len(preds) if preds else float("nan")


def _reparse(rows, raw_key: str, scheme):
    """Re-derive S2 predictions from stored raw outputs.

    Re-parsing rather than trusting the stored ``pred_*`` keeps every cell on
    one parser version; the stored values were written by several. Cells that
    kept no raw text fall back to the prediction they recorded, and an item the
    run excluded (a content-filter refusal) is not an answer to stratify.
    ``answer_idx == -1`` marks a truly-Unknown item, a different subset.
    """
    out = []
    for r in rows:
        if r.get("answer_idx", 0) < 0 or r.get("excluded"):
            continue
        raw = r.get(raw_key)
        if raw is None:
            pred = r.get("pred") or r.get("pred_s2")
            if pred is None:
                continue
        else:
            pred, _ = EV.parse_judge_tiered(raw, scheme, with_unknown=True)
        out.append((r["id"], pred))
    return out


#: A local temperature cell must hold this many items to be plotted.
LOCAL_TEMP_EXPECT_N = 500


def _local_temperature_table():
    """({display: {dataset: [Abs Rate % per T]}}, coverage rows) for locally
    served checkpoints.

    Same metric and the same parser as the API sweep; only the file layout
    differs. Two things are verified before a value is allowed onto the curve,
    because a point that is merely present looks exactly like a point that is
    right:

    * the item count matches :data:`LOCAL_TEMP_EXPECT_N`, so a short or
      partly-written cell is dropped rather than plotted beside full ones;
    * the temperature the run recorded matches the one the filename claims, so
      a mis-named file cannot land on the wrong x position.

    A cell failing either test becomes NaN and is reported, which plots as a
    gap. A missing cell does the same, so a sweep in flight shows gaps rather
    than a curve that silently shifts as jobs land.
    """
    table, coverage, problems, budgets = {}, [], [], {}
    for disp, (d, tag) in LOCAL_TEMP_DIRS.items():
        if not d.exists():
            continue
        per_ds = {}
        for ds in ("FLD", "FOLIO"):
            scheme = get_scheme(ds)
            row, n_ok = [], 0
            for t in TEMPS:
                suffix = f"_T{t}".replace(".", "p")
                f = d / f"ab_summary_{ds}_{tag}{suffix}.json"
                if not f.exists():
                    row.append(float("nan"))
                    continue
                cell = json.loads(f.read_text())
                rows = cell["per_sample"]
                cfg = cell.get("run_config") or {}
                if "temperature" not in cfg:
                    # Fail closed. Skipping the check when the field is absent
                    # would wave through exactly the files it exists for: any
                    # written by a runner version that did not record it, whose
                    # temperature is then unknowable from the file itself.
                    problems.append(f"{f.name}: no run_config.temperature, so the "
                                    f"setting it ran at cannot be verified")
                    row.append(float("nan")); continue
                recorded = cfg["temperature"]
                if recorded is None or abs(float(recorded) - t) > 1e-9:
                    problems.append(f"{f.name}: recorded T={recorded}, name says {t}")
                    row.append(float("nan")); continue
                if len(rows) != LOCAL_TEMP_EXPECT_N:
                    problems.append(f"{f.name}: {len(rows)} items, "
                                    f"expected {LOCAL_TEMP_EXPECT_N}")
                    row.append(float("nan")); continue
                # Every point on one curve must share a generation budget. The
                # cap moves Abs Rate by tens of points -- raising it from 1024
                # to 3072 took one cell from 13% to 46% -- so a re-run left
                # beside the cells it replaces would look like a temperature
                # effect. Recorded per cell, compared across the sweep below.
                if "max_new_tokens" not in cfg:
                    problems.append(f"{f.name}: no run_config.max_new_tokens, so "
                                    f"the generation budget cannot be verified")
                    row.append(float("nan")); continue
                budgets.setdefault((disp, ds), {})[t] = cfg["max_new_tokens"]
                row.append(_abs_rate([p_ for _, p_ in _reparse(rows, "raw_s2", scheme)]))
                n_ok += 1
            # Emit the row even when every cell was rejected. Dropping the
            # key instead would leave a model present in the table with one
            # dataset missing, and a consumer indexing both would raise rather
            # than draw the gap the rejections are supposed to show.
            seen_budgets = set(budgets.get((disp, ds), {}).values())
            if len(seen_budgets) > 1:
                # Mixed budgets across one curve: drop the whole series rather
                # than let the reader compare points that are not comparable.
                per_temp = budgets[(disp, ds)]
                problems.append(
                    f"{disp}/{ds}: cells were generated at different "
                    f"max_new_tokens ({', '.join(f'T{k}={v}' for k, v in sorted(per_temp.items()))}) "
                    f"-- the whole series is dropped; keep one budget per sweep")
                row = [float("nan")] * len(TEMPS)
                n_ok = 0
            per_ds[ds] = [round(v, 1) if v == v else v for v in row]
            coverage.append((disp, ds, n_ok * LOCAL_TEMP_EXPECT_N,
                             len(TEMPS) * LOCAL_TEMP_EXPECT_N))
        if any(v == v for row in per_ds.values() for v in row):
            table[disp] = per_ds
    for msg in problems:
        print(f"  [warn] dropped {msg}")
    return table, coverage


def temperature_table():
    """{display: {dataset: [Abs Rate % per T]}}, plus a T=0 coverage report."""
    table, coverage = {}, []
    for disp, tag in MODELS.items():
        if tag not in TEMPERATURE_HONOURED:
            continue
        table[disp] = {}
        for ds in ("FLD", "FOLIO"):
            scheme = get_scheme(ds)
            row, sweep_ids = [], None
            for t in TEMPS:
                if t == 0.0:
                    row.append(None)
                    continue
                p = SWEEP_DIR / f"summary_{_TEMP_TAGS[t]}_{ds}_{tag}.json"
                if not p.exists():
                    row.append(float("nan"))
                    continue
                rows = json.loads(p.read_text())["per_sample"]
                # Pool the complement pass so the cell reports the n=500 the
                # paper does, not the 200 the first sweep drew.
                p2 = SWEEP_P2_DIR / p.name
                if p2.exists():
                    have = {r["id"] for r in rows}
                    p2_rows = json.loads(p2.read_text())["per_sample"]
                    extra = [r for r in p2_rows if r["id"] not in have]
                    if len(extra) != len(p2_rows):
                        raise SystemExit(
                            f"{p2.name}: {len(p2_rows) - len(extra)} item(s) also "
                            f"appear in the first sweep. The passes must be "
                            f"complementary or the pooled cell double-counts.")
                    rows = rows + extra
                sweep_ids = sweep_ids or {r["id"] for r in rows}
                row.append(_abs_rate([p_ for _, p_ in _reparse(rows, "raw", scheme)]))

            base, seen = [], set()
            # The batch files only hold the original 200; once the sweep is
            # pooled to 500, the n=500 main runs are needed for the rest of the
            # T=0 baseline.
            sources = [RESULTS / rel for rel in BASELINE_SOURCES[(tag, ds)]]
            if sweep_ids and len(sweep_ids) > 200:
                sources += sorted(RESULTS.glob(f"ab_*/ab_summary_{ds}*_{tag}.json"))
            for path in sources:
                if not path.exists():
                    continue
                for r in load_summary(path).get("per_sample") or []:
                    rid = r["id"].replace("FLD500_", "FLD_")   # main run names FLD items differently
                    if rid in (sweep_ids or ()) and rid not in seen and "raw_s2" in r:
                        seen.add(rid)
                        base.append(r)
            base_preds = [p_ for _, p_ in _reparse(base, "raw_s2", scheme)]
            # A T=0 cell run alongside the complement covers items the main runs
            # never touched (FOLIO's main run stops at n=300), and it is the same
            # code path as the other temperatures rather than a different one.
            t0 = SWEEP_P2_DIR / f"summary_T0p0_{ds}_{tag}.json"
            n_base = len(base)
            if t0.exists():
                have = {r["id"] for r in base}
                extra = [r for r in json.loads(t0.read_text())["per_sample"]
                         if r["id"] not in have]
                base_preds += [p_ for _, p_ in _reparse(extra, "raw", scheme)]
                n_base += len(extra)
            row[0] = _abs_rate(base_preds)
            coverage.append((disp.replace("\n", " "), ds, n_base, len(sweep_ids or ())))
            table[disp][ds] = [round(v, 1) if v == v else v for v in row]
    local_table, local_cov = _local_temperature_table()
    table.update(local_table)
    coverage.extend(local_cov)
    return table, coverage


# ----------------------------------------------------------------------
# S10(b) difficulty
# ----------------------------------------------------------------------
# dataset/FLD.json carries proof depths 1-8, ~62 items each -- one bin per
# depth, not the 1-3 / 4-6 / ... / 16+ bins the figure used to hardcode. Those
# ran to "16+" with non-zero values, which this dataset cannot produce.
BINS = [str(d) for d in range(1, 9)]


def _norm_id(sid: str) -> str:
    """Result files name FLD items ``FLD_0007`` or ``FLD500_0007``; the dataset
    uses the former."""
    return sid.replace("FLD500_", "FLD_") if isinstance(sid, str) else sid


def _fld_depths():
    out = {}
    for it in json.loads((ROOT / "dataset" / "FLD.json").read_text()):
        if it.get("depth") is not None and "id" in it:
            out[it["id"]] = int(it["depth"])
    return out


def difficulty_table():
    """{display: [Abs Rate % per depth bin]}, per-bin n, and Spearman rho.

    Two rho's, because only one is comparable to the paper's:
      * ``rho_item`` -- over the (depth, abstained 0/1) item pairs. This is the
        item-level correlation the paper reports.
      * ``rho_bin``  -- over the 8 bin means. Always far larger, because
        binning averages the item-level noise away.
    """
    depths = _fld_depths()
    scheme = get_scheme("FLD")
    table, counts, rhos = {}, {}, {}
    for disp, tag in MODELS.items():
        per_bin, items = defaultdict(list), []
        path = RESULTS / DIFFICULTY_SOURCE[tag]
        if path.exists():
            for sid, pred in _reparse(load_summary(path).get("per_sample") or [],
                                      "raw_s2", scheme):
                d = depths.get(_norm_id(sid))
                if d is None or not 1 <= d <= 8:
                    continue
                per_bin[d - 1].append(pred)
                items.append((d, 1 if pred == "UNKNOWN" else 0))
        table[disp] = [_abs_rate(per_bin[i]) if per_bin[i] else float("nan")
                       for i in range(len(BINS))]
        counts[disp] = [len(per_bin[i]) for i in range(len(BINS))]
        rhos[disp] = _spearman(items, table[disp])
    return table, counts, rhos


def _spearman(items, bin_means):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return {}
    out = {}
    if items:
        rho, p = spearmanr([d for d, _ in items], [a for _, a in items])
        out.update(rho_item=float(rho), p_item=float(p), n_item=len(items))
    xs = [i for i, v in enumerate(bin_means) if v == v]
    if len(xs) > 2:
        rho, p = spearmanr(xs, [bin_means[i] for i in xs])
        out.update(rho_bin=float(rho), p_bin=float(p))
    return out


if __name__ == "__main__":
    t, cov = temperature_table()
    # A model read through LOCAL_TEMP_DIRS is plotted, not dropped -- listing it
    # as excluded because its API tag is missing from TEMPERATURE_HONOURED
    # would contradict the curve printed right below.
    dropped = [d.replace(chr(10), " ") for d, g in MODELS.items()
               if g not in TEMPERATURE_HONOURED
               and d not in LOCAL_TEMP_DIRS]
    print("=== S10(a) Temperature — Abs Rate (%) ===")
    if dropped:
        print(f"  excluded (endpoint does not apply the temperature setting): "
              f"{', '.join(dropped)}")
    for disp, per_ds in t.items():
        for ds, row in per_ds.items():
            print(f"  {disp.replace(chr(10),' '):<24} {ds:<6} " +
                  "  ".join(f"{v:5.1f}" for v in row))
    print("\n  T=0 baseline matched / sweep items:")
    for disp, ds, m, n in cov:
        print(f"    {disp:<24} {ds:<6} {m}/{n}" + ("" if m == n else "   <-- INCOMPLETE"))

    d, c, rhos = difficulty_table()
    print("\n=== S10(b) Difficulty — Abs Rate (%) by proof-step count ===")
    print(f"  {'model':<24} " + "  ".join(f"{b:>6}" for b in BINS))
    for disp, row in d.items():
        print(f"  {disp.replace(chr(10),' '):<24} " +
              "  ".join(f"{v:6.1f}" if v == v else "     —" for v in row))
    print(f"  {'n=':<24} " + "  ".join(f"{v:>6}" for v in next(iter(c.values()))))
    print("\n  Spearman rho(depth, abstained):")
    for disp, r in rhos.items():
        if r:
            print(f"    {disp.replace(chr(10),' '):<24} "
                  f"item rho={r['rho_item']:+.3f} (p={r['p_item']:.2g}, n={r['n_item']})   "
                  f"bin rho={r.get('rho_bin', float('nan')):+.3f}")
