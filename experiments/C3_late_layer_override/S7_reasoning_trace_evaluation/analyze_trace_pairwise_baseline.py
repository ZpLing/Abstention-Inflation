#!/usr/bin/env python3
"""Pairwise trace similarity with noise-floor baselines.

For each of 6 cells (3 models x {FLD, FOLIO}):
  - F1(S1, S2)              pairwise (same sample, different prompt)   ← the
                            metric C0 *should* be reporting
  - F1(S2_T=0, S2_T=0.3)    pairwise (same prompt, sampling noise)     ← noise floor
  - F1(S1[i], S1[j!=i])     random cross-sample pairing                ← random floor
  - F1(S1, gold)            for reference (the current C0 number)
  - F1(S2, gold)            for reference (the current C0 number)

FLD  -> set-F1 over {fact_i, int_i} atom references
FOLIO -> BERTScore-F1 over free text (roberta-large, rescale_with_baseline)
"""
import json
import os
import random
import sys

sys.path.insert(0, '.')

from core.metrics import trace_set_f1, _bertscore_pair
from core.trace_extractors import _extract_formal_refs

ROOT = 'results'
DATA_ROOT = 'dataset'

CELLS = [
    # (dataset, model, [batch dirs for S1+S2], temp_sweep_filename_model_tag)
    ('FLD',   'deepseek-v4-flash', ['ab_e_option_baseline', 'ab_deepseek_batch2']),
    ('FOLIO', 'deepseek-v4-flash', ['ab_followup',          'ab_deepseek_batch2']),
    ('FLD',   'gpt-5.4-nano',                 ['ab_gpt5_nano',         'ab_nano_batch2']),
    ('FOLIO', 'gpt-5.4-nano',                 ['ab_gpt5_nano',         'ab_nano_batch2']),
    ('FLD',   'gemini-3.1-flash-lite',        ['ab_gemini_flash_lite', 'ab_gemini_batch2']),
    ('FOLIO', 'gemini-3.1-flash-lite',        ['ab_gemini_flash_lite', 'ab_gemini_batch2']),
]


def load_ab(ds, model, dirs):
    """Return {id: {raw_s1, raw_s2}} from all batch dirs combined."""
    out = {}
    for d in dirs:
        path = f'{ROOT}/{d}/ab_summary_{ds}_{model}.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data.get('per_sample', []):
            sid = r.get('id')
            if sid is None:
                continue
            out[sid] = {
                'raw_s1': r.get('raw_s1') or '',
                'raw_s2': r.get('raw_s2') or '',
            }
    return out


def load_temp_sweep(ds, model, temp_tag='T0p3'):
    """Return {id: raw} for the temperature-sweep S2 traces."""
    path = f'{ROOT}/temperature_sweep/summary_{temp_tag}_{ds}_{model}.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for r in data.get('per_sample', []):
        sid = r.get('id')
        if sid is None:
            continue
        out[sid] = r.get('raw') or ''
    return out


def load_gold(ds):
    """Return {id: gold_proof_str | gold_steps_text}.

    FLD: id -> joined 'gold_proof_steps' for fact/int regex extraction
    FOLIO: id -> joined derivation prose
    """
    path = f'{DATA_ROOT}/{ds}.json'
    if not os.path.exists(path):
        print(f'  ! gold not found: {path}')
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for item in data:
        sid = item.get('id')
        if sid is None:
            continue
        if ds.upper().startswith('FLD'):
            steps = item.get('gold_proof_steps') or []
            if isinstance(steps, list):
                out[sid] = ' '.join(str(s) for s in steps)
            else:
                out[sid] = str(steps)
        elif ds.upper().startswith('FOLIO'):
            steps = item.get('gold_proof_steps') or []
            parts = []
            for st in steps:
                if isinstance(st, dict):
                    t = (st.get('derivation') or '').strip().rstrip('.')
                    if t:
                        parts.append(t)
            out[sid] = '. '.join(parts)
    return out


def fld_pair_set_f1(a, b):
    """Pairwise set-F1 between two FLD traces (no gold)."""
    p = _extract_formal_refs(a)
    g = _extract_formal_refs(b)
    return trace_set_f1(p, g)[2]


def fld_vs_gold_set_f1(trace, gold_text):
    p = _extract_formal_refs(trace)
    g = _extract_formal_refs(gold_text)
    return trace_set_f1(p, g)[2]


def folio_pairwise_bertscore(pairs):
    """pairs: list of (cand, ref). Returns list of F1 (0.0 for empty pairs)."""
    idx = [i for i, (a, b) in enumerate(pairs) if a and b]
    cands = [pairs[i][0] for i in idx]
    refs  = [pairs[i][1] for i in idx]
    if not cands:
        return [0.0] * len(pairs)
    f1s = _bertscore_pair(cands, refs)
    out = [0.0] * len(pairs)
    for j, i in enumerate(idx):
        out[i] = f1s[j]
    return out


def main():
    random.seed(42)
    rows = []
    for ds, model, dirs in CELLS:
        print(f'\n=== {ds} / {model} ===')
        ab = load_ab(ds, model, dirs)
        ts = load_temp_sweep(ds, model, 'T0p3')
        gold = load_gold(ds)
        # intersect ids
        common = sorted(set(ab.keys()) & set(ts.keys()))
        common_gold = sorted(set(ab.keys()) & set(gold.keys()))
        print(f'  n_ab={len(ab)}, n_T0p3={len(ts)}, n_gold={len(gold)}, '
              f'common(ab∩T0p3)={len(common)}, common(ab∩gold)={len(common_gold)}')

        # ───── Pair lists ─────
        # 1) S1 vs S2 (pairwise, on samples present in ab)
        s1_s2_pairs = [(ab[i]['raw_s1'], ab[i]['raw_s2']) for i in sorted(ab.keys())]
        # 2) Noise floor: S2 T=0 vs S2 T=0.3 (on samples present in both)
        noise_pairs = [(ab[i]['raw_s2'], ts[i]) for i in common]
        # 3) Random cross-sample: S1[i] vs S1[π(i)] permutation
        s1_only = [ab[i]['raw_s1'] for i in sorted(ab.keys())]
        perm = list(range(len(s1_only)))
        random.shuffle(perm)
        # Ensure no fixed points
        while any(perm[k] == k for k in range(len(perm))):
            random.shuffle(perm)
        cross_pairs = [(s1_only[k], s1_only[perm[k]]) for k in range(len(s1_only))]
        # 4) S1 vs gold and S2 vs gold (reference numbers)
        s1_gold = [(ab[i]['raw_s1'], gold[i]) for i in common_gold]
        s2_gold = [(ab[i]['raw_s2'], gold[i]) for i in common_gold]

        if ds.upper().startswith('FLD'):
            f1_s1_s2  = [fld_pair_set_f1(a, b) for a, b in s1_s2_pairs]
            f1_noise  = [fld_pair_set_f1(a, b) for a, b in noise_pairs]
            f1_cross  = [fld_pair_set_f1(a, b) for a, b in cross_pairs]
            f1_s1g    = [fld_vs_gold_set_f1(a, b) for a, b in s1_gold]
            f1_s2g    = [fld_vs_gold_set_f1(a, b) for a, b in s2_gold]
            metric = 'set-F1'
        else:  # FOLIO
            print('  computing BERTScore (4 batches)...')
            f1_s1_s2 = folio_pairwise_bertscore(s1_s2_pairs)
            f1_noise = folio_pairwise_bertscore(noise_pairs)
            f1_cross = folio_pairwise_bertscore(cross_pairs)
            f1_s1g   = folio_pairwise_bertscore(s1_gold)
            f1_s2g   = folio_pairwise_bertscore(s2_gold)
            metric = 'BERTScore'

        def m(xs):
            return sum(xs) / len(xs) if xs else 0.0

        row = {
            'cell': f'{ds}/{model}',
            'metric': metric,
            'n_pair_S1_S2':   len(s1_s2_pairs),
            'n_pair_noise':   len(noise_pairs),
            'n_pair_cross':   len(cross_pairs),
            'n_pair_gold':    len(s1_gold),
            'F1_S1_S2':       m(f1_s1_s2),
            'F1_noise':       m(f1_noise),
            'F1_cross':       m(f1_cross),
            'F1_S1_gold':     m(f1_s1g),
            'F1_S2_gold':     m(f1_s2g),
        }
        rows.append(row)
        print(f'  {metric}:')
        print(f'    F1(S1, S2)              = {row["F1_S1_S2"]:.3f}  (n={row["n_pair_S1_S2"]})')
        print(f'    F1(S2_T=0, S2_T=0.3)    = {row["F1_noise"]:.3f}  (n={row["n_pair_noise"]})')
        print(f'    F1(S1[i], S1[j!=i])     = {row["F1_cross"]:.3f}  (n={row["n_pair_cross"]})')
        print(f'    F1(S1, gold)            = {row["F1_S1_gold"]:.3f}  (n={row["n_pair_gold"]})')
        print(f'    F1(S2, gold)            = {row["F1_S2_gold"]:.3f}  (n={row["n_pair_gold"]})')

    # Final table
    print('\n' + '=' * 110)
    print(f'{"cell":40s} | {"metric":10s} | F1(S1,S2) | noise   | random  | S1,gold | S2,gold')
    print('-' * 110)
    for r in rows:
        print(f'{r["cell"]:40s} | {r["metric"]:10s} | {r["F1_S1_S2"]:9.3f} | '
              f'{r["F1_noise"]:7.3f} | {r["F1_cross"]:7.3f} | '
              f'{r["F1_S1_gold"]:7.3f} | {r["F1_S2_gold"]:7.3f}')

    out_path = 'results/analysis/trace_pairwise_baseline.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
