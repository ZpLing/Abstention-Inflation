#!/usr/bin/env python3
"""Step-level embedding similarity for reasoning traces.

For each of 6 cells (3 models x {FLD, FOLIO}):
  - Split each trace into reasoning steps (sentence-level, filter trivials)
  - Embed every step with text-embedding-3-large (API gateway)
  - Compute step-F1 (BERTScore-style at step granularity):
      step-P = mean over S1 steps of max cosine with any S2 step
      step-R = mean over S2 steps of max cosine with any S1 step
      step-F1 = 2PR/(P+R)
  - Three comparisons:
      * F1(S1, S2)
      * F1(S2_T=0, S2_T=0.3)      [noise floor]
      * F1(S1[i], S1[π(i)])        [random cross-sample floor]

Embeddings are cached on disk per (cell, source-tag, sample-id).
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import yaml

ROOT = 'results'
CACHE_DIR = Path('results/analysis/step_embedding_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = 'results/analysis/trace_step_embedding.json'

with open('./config') as f:
    CFG = yaml.safe_load(f)
API_KEY = CFG['llm']['api_key']
BASE_URL = CFG['llm']['base_url']
EMBED_MODEL = 'text-embedding-3-large'
MAX_BATCH = 8           # small batch → fewer timeouts on slow gateway
MAX_INPUT_CHARS = 4000  # ≈ 1k tokens; long sentences truncated
HTTP_TIMEOUT = 30       # 30s per call
N_WORKERS = 200         # parallel traces (user-approved)

CELLS = [
    ('FLD',   'deepseek-r1-distill-llama-8b', ['ab_e_option_baseline', 'ab_deepseek_batch2']),
    ('FOLIO', 'deepseek-r1-distill-llama-8b', ['ab_followup',          'ab_deepseek_batch2']),
    ('FLD',   'gpt-5.4-nano',                 ['ab_gpt5_nano',         'ab_nano_batch2']),
    ('FOLIO', 'gpt-5.4-nano',                 ['ab_gpt5_nano',         'ab_nano_batch2']),
    ('FLD',   'gemini-3.1-flash-lite',        ['ab_gemini_flash_lite', 'ab_gemini_batch2']),
    ('FOLIO', 'gemini-3.1-flash-lite',        ['ab_gemini_flash_lite', 'ab_gemini_batch2']),
]

# ───────────── step splitting ─────────────
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')
NUMBERED_STEP_RE = re.compile(r'(?m)^\s*(?:step\s*)?(\d+)[.)\]]\s+', re.IGNORECASE)


def split_steps(text: str, min_tokens: int = 5) -> list:
    """Return reasoning steps. Prefer numbered list if present, else sentences."""
    if not text:
        return []
    # Try numbered-step parsing first
    matches = list(NUMBERED_STEP_RE.finditer(text))
    if len(matches) >= 3:  # use numbered split only if there are ≥3 numbered items
        pieces = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[m.start():end].strip()
            if chunk:
                pieces.append(chunk)
    else:
        pieces = SENT_SPLIT_RE.split(text)

    out = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(p.split()) < min_tokens:
            continue
        if len(p) > MAX_INPUT_CHARS:
            p = p[:MAX_INPUT_CHARS]
        out.append(p)
    return out


# ───────────── embedding ─────────────
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=N_WORKERS, pool_maxsize=N_WORKERS)
_session.mount('https://', _adapter)
_session.mount('http://',  _adapter)


def _embed_batch(texts: list, max_retries: int = 3) -> np.ndarray:
    payload = {'model': EMBED_MODEL, 'input': texts}
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    for attempt in range(max_retries):
        try:
            r = _session.post(f'{BASE_URL}/embeddings', headers=headers, json=payload, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                vecs = np.array([d['embedding'] for d in j['data']], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return vecs / norms
            wait = min(2 ** attempt, 8)
            time.sleep(wait)
        except requests.RequestException:
            wait = min(2 ** attempt, 8)
            time.sleep(wait)
    raise RuntimeError(f'embedding failed after {max_retries} retries')


def embed_with_cache(trace_id: str, steps: list) -> np.ndarray:
    """Returns (n_steps, dim) array. Cached by trace_id."""
    if not steps:
        return np.zeros((0, 3072), dtype=np.float32)
    cache_key = hashlib.md5(f'{trace_id}::{len(steps)}::{steps[0][:50] if steps else ""}'.encode()).hexdigest()
    cache_file = CACHE_DIR / f'{cache_key}.npy'
    if cache_file.exists():
        return np.load(cache_file)
    # Batch
    all_vecs = []
    for i in range(0, len(steps), MAX_BATCH):
        batch = steps[i:i + MAX_BATCH]
        vecs = _embed_batch(batch)
        all_vecs.append(vecs)
    arr = np.vstack(all_vecs)
    np.save(cache_file, arr)
    return arr


# ───────────── similarity ─────────────
def step_f1(A: np.ndarray, B: np.ndarray) -> dict:
    """BERTScore-style step F1.

    A, B are (n, dim) L2-normalized step embeddings.
    Returns dict with P, R, F1, n_A, n_B.
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return {'P': 0.0, 'R': 0.0, 'F1': 0.0, 'n_A': int(A.shape[0]), 'n_B': int(B.shape[0])}
    # cosine matrix (already L2-normalized → dot product = cosine)
    M = A @ B.T   # (n_A, n_B)
    # Precision: each A finds best B
    P = float(M.max(axis=1).mean())
    # Recall: each B finds best A
    R = float(M.max(axis=0).mean())
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {'P': P, 'R': R, 'F1': F, 'n_A': int(A.shape[0]), 'n_B': int(B.shape[0])}


# ───────────── data loading ─────────────
def load_ab(ds, model, dirs):
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
            out[sid] = {'raw_s1': r.get('raw_s1') or '', 'raw_s2': r.get('raw_s2') or ''}
    return out


def load_temp_sweep(ds, model, tag='T0p3'):
    path = f'{ROOT}/temperature_sweep/summary_{tag}_{ds}_{model}.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r['id']: (r.get('raw') or '') for r in data.get('per_sample', []) if r.get('id')}


# ───────────── main ─────────────
def run_cell(ds, model, dirs):
    print(f'\n=== {ds} / {model} ===', flush=True)
    ab = load_ab(ds, model, dirs)
    ts = load_temp_sweep(ds, model, 'T0p3')
    ids_all = sorted(ab.keys())
    ids_noise = sorted(set(ab.keys()) & set(ts.keys()))
    print(f'  n_ab={len(ab)}, n_noise_overlap={len(ids_noise)}', flush=True)

    # Step-split all needed traces
    cell_tag = f'{ds}__{model}'
    s1_steps = {sid: split_steps(ab[sid]['raw_s1']) for sid in ids_all}
    s2_steps = {sid: split_steps(ab[sid]['raw_s2']) for sid in ids_all}
    ns_steps = {sid: split_steps(ts[sid]) for sid in ids_noise}

    # Report avg step counts
    avg_s1 = np.mean([len(v) for v in s1_steps.values()]) if s1_steps else 0
    avg_s2 = np.mean([len(v) for v in s2_steps.values()]) if s2_steps else 0
    avg_ns = np.mean([len(v) for v in ns_steps.values()]) if ns_steps else 0
    total_steps = sum(len(v) for v in s1_steps.values()) + sum(len(v) for v in s2_steps.values()) + sum(len(v) for v in ns_steps.values())
    print(f'  avg steps S1={avg_s1:.1f}  S2={avg_s2:.1f}  noise(T0.3)={avg_ns:.1f}  total_step_embeds={total_steps}', flush=True)

    # Embed — parallel across all (source, sid) trace jobs
    print(f'  embedding (parallel={N_WORKERS}) ...', flush=True)
    t0 = time.time()
    jobs = []
    for sid in ids_all:
        jobs.append(('S1', sid, s1_steps[sid]))
        jobs.append(('S2', sid, s2_steps[sid]))
    for sid in ids_noise:
        jobs.append(('NS_T0p3', sid, ns_steps[sid]))
    print(f'  total trace jobs = {len(jobs)}', flush=True)

    results = {}
    done_count = [0]

    def _work(job):
        src, sid, steps = job
        emb = embed_with_cache(f'{cell_tag}::{src}::{sid}', steps)
        return (src, sid), emb

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_work, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                key, emb = fut.result()
                results[key] = emb
            except Exception as e:
                print(f'    ! job failed: {e}', flush=True)
            done_count[0] += 1
            if done_count[0] % 100 == 0 or done_count[0] == len(jobs):
                print(f'    [{done_count[0]}/{len(jobs)}] elapsed {time.time()-t0:.1f}s', flush=True)

    s1_emb = {sid: results.get(('S1', sid), np.zeros((0, 3072), dtype=np.float32)) for sid in ids_all}
    s2_emb = {sid: results.get(('S2', sid), np.zeros((0, 3072), dtype=np.float32)) for sid in ids_all}
    ns_emb = {sid: results.get(('NS_T0p3', sid), np.zeros((0, 3072), dtype=np.float32)) for sid in ids_noise}
    print(f'  embeddings done in {time.time()-t0:.1f}s', flush=True)

    # Comparisons
    f1_s1_s2 = []
    for sid in ids_all:
        r = step_f1(s1_emb[sid], s2_emb[sid])
        f1_s1_s2.append(r['F1'])
    f1_noise = []
    for sid in ids_noise:
        r = step_f1(s2_emb[sid], ns_emb[sid])
        f1_noise.append(r['F1'])
    # Random cross-sample (permutation of S1 indices)
    perm = list(range(len(ids_all)))
    random.seed(42)
    random.shuffle(perm)
    while any(perm[k] == k for k in range(len(perm))):
        random.shuffle(perm)
    f1_cross = []
    for k, sid in enumerate(ids_all):
        sid_other = ids_all[perm[k]]
        r = step_f1(s1_emb[sid], s1_emb[sid_other])
        f1_cross.append(r['F1'])

    def m(xs): return float(np.mean(xs)) if xs else 0.0
    def s(xs): return float(np.std(xs)) if xs else 0.0

    row = {
        'cell': f'{ds}/{model}',
        'n_S1_S2': len(f1_s1_s2),
        'n_noise': len(f1_noise),
        'n_cross': len(f1_cross),
        'avg_steps_S1': float(avg_s1),
        'avg_steps_S2': float(avg_s2),
        'avg_steps_noise': float(avg_ns),
        'F1_S1_S2': m(f1_s1_s2),  'std_S1_S2': s(f1_s1_s2),
        'F1_noise': m(f1_noise),  'std_noise': s(f1_noise),
        'F1_cross': m(f1_cross),  'std_cross': s(f1_cross),
    }
    print(f'  step-F1(S1, S2)              = {row["F1_S1_S2"]:.3f}  ± {row["std_S1_S2"]:.3f}', flush=True)
    print(f'  step-F1(S2_T=0, S2_T=0.3)    = {row["F1_noise"]:.3f}  ± {row["std_noise"]:.3f}', flush=True)
    print(f'  step-F1(S1[i], S1[j!=i])     = {row["F1_cross"]:.3f}  ± {row["std_cross"]:.3f}', flush=True)
    return row


def main():
    rows = []
    for ds, model, dirs in CELLS:
        try:
            rows.append(run_cell(ds, model, dirs))
        except Exception as e:
            print(f'  ! cell failed: {e}')
            import traceback; traceback.print_exc()
        # Persist after every cell
        with open(OUT_PATH, 'w') as f:
            json.dump(rows, f, indent=2)
    # Final table
    print('\n' + '=' * 120)
    print(f'{"cell":40s} | {"steps S1/S2/N":15s} | F1(S1,S2)    | F1(noise)    | F1(random)')
    print('-' * 120)
    for r in rows:
        print(f'{r["cell"]:40s} | '
              f'{r["avg_steps_S1"]:4.1f}/{r["avg_steps_S2"]:4.1f}/{r["avg_steps_noise"]:4.1f}  | '
              f'{r["F1_S1_S2"]:.3f} ± {r["std_S1_S2"]:.3f} | '
              f'{r["F1_noise"]:.3f} ± {r["std_noise"]:.3f} | '
              f'{r["F1_cross"]:.3f} ± {r["std_cross"]:.3f}')
    print(f'\nSaved → {OUT_PATH}')


if __name__ == '__main__':
    main()
