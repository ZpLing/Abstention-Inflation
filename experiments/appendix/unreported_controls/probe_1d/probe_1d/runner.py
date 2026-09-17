"""Probe-1D top-level orchestrator.

Reads existing main / supplementary summaries, extracts hidden states from
a local HF model, runs P1→P4 in sequence, and writes a single JSON summary.

Inputs (all on disk; no API calls):
    results/ab/ab_summary_<dataset>_<probed_model_name>.json
        — main Exp 1 product. We use it to identify:
            * Abstention Inflation samples           (S2 output Unknown)
            * S1-correct samples    (γ training pool + P2 training set)
            * S1-incorrect samples  (β training pool)
        Also gives the gold answer_idx for each sample.
    results/supplementary/supp_summary_FLD_<probed_model_name>.json
        — α training pool: FLD genuinely-Unknown samples whose S2 output Unknown.

Hidden-state extraction is then done by re-running S1 / S2 prompts through a
*local* HF model (the probed model). The probed model's name in the summary
files is decoupled from the local checkpoint path: closed-API runs may have
been done against e.g. "gpt-4o" while the probe is run against a smaller
local model that supports hidden-state access. This decoupling matches
PLAN_v2 §1D's note that the probe is on small models only.

When the probed-model name in the summaries matches the local checkpoint
(common case), set probe_1d.summary_model_name = probe_1d.local_model_name
and the runner reads the same row of summaries it used for behavioral
comparison.

Config block (configs/experiment.yaml):

    probe_1d:
      datasets:               ["MedQA"]
      ab_results_dir:         "results/ab"
      supp_results_dir:       "results/supplementary"
      results_dir:            "results/supplementary"

      summary_model_name:     "gemma-4-E2B-it"      # which ab_summary to read
      local_model_path:       "models/gemma-4-E2B-it"
      probe_label:            "gemma-4-E2B-it"      # used in output filename

      device:                 null                  # auto: mps/cuda/cpu
      dtype:                  "float16"
      batch_size:             4
      max_seq_length:         2048
      max_ai_samples:        100                   # cap Abstention Inflation set; 0 = no cap
      max_train_pool_each:    200                   # cap γ / β / α pool sizes
      run_p4:                 true                  # set false to skip ablation
      p4_max_new_tokens:      200

The summary file is `results/supplementary/probe_1d_<dataset>_<probe_label>.json`.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.label_scheme import get_scheme
from core.prompts import (
    build_mcq_s1_prompt, build_mcq_s2_prompt,
    build_judge_s1_prompt, build_judge_s2_prompt,
)

from core.data_handler import DataHandler

from . import hidden_states as hs_mod
from .p1_aid import fit_p1, P1Result
from .p2_letter_probe import fit_p2, P2Result
from .p3_attribution import fit_p3, P3Result
from .p4_ablation import run_p4, P4Result
from core.result_schema import get_field


_LETTERS = ["A", "B", "C", "D"]


class Probe1DRunner:
    def __init__(self, config: Dict[str, Any], data_handler: DataHandler):
        self.config = config
        self.data_handler = data_handler

        cfg = config.get("probe_1d", {})
        self.dataset_names: List[str] = cfg.get("datasets", ["MedQA"])
        self.ab_results_dir = Path(cfg.get("ab_results_dir", "results/ab"))
        self.supp_results_dir = Path(cfg.get("supp_results_dir", "results/supplementary"))
        self.results_dir = Path(cfg.get("results_dir", "results/supplementary"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.summary_model_name = cfg.get("summary_model_name") or cfg.get("probe_label")
        self.local_model_path = cfg["local_model_path"]
        self.probe_label = (
            cfg.get("probe_label")
            or Path(self.local_model_path).name
        )

        self.device = cfg.get("device")
        self.dtype = cfg.get("dtype", "float16")
        self.batch_size = int(cfg.get("batch_size", 4))
        self.max_seq_length = int(cfg.get("max_seq_length", 2048))
        self.max_ai_samples = int(cfg.get("max_ai_samples", 100))
        self.max_train_pool_each = int(cfg.get("max_train_pool_each", 200))
        self.run_p4_flag = bool(cfg.get("run_p4", True))
        self.p4_max_new_tokens = int(cfg.get("p4_max_new_tokens", 200))

        self._model = None
        self._tokenizer = None
        self._device = None

    # =================================================================
    # Top-level entry
    # =================================================================
    def run(self):
        if not self.dataset_names:
            print("[Probe1DRunner] No datasets configured — nothing to do.")
            return
        for ds_name in self.dataset_names:
            print(f"\n===== Probe 1D :: {ds_name} =====")
            self._run_one(ds_name)

    # =================================================================
    # Single-dataset pipeline
    # =================================================================
    def _run_one(self, ds_name: str):
        ab_summary = self._read_ab_summary(ds_name)
        if ab_summary is None:
            return

        # Categorize samples by S1/S2 outcome.
        per_sample = ab_summary["per_sample"]
        s5_rerun = {
            r["sample_id"]: r for r in get_field(ab_summary, "s5_rerun", [])
        }

        # γ-pool: S1 == correct letter.
        gamma_rows = [r for r in per_sample if _is_correct(r["pred_s1"], r["answer_idx"])]
        # β-pool: S1 was a concrete answer (A-D) but wrong.
        beta_rows = [
            r for r in per_sample
            if r["pred_s1"] in _LETTERS and not _is_correct(r["pred_s1"], r["answer_idx"])
        ]
        # Abs Rate: S2 == UNKNOWN.
        ai_rows = [r for r in per_sample if r["pred_s2"] == "UNKNOWN"]

        print(f"  pools: γ={len(gamma_rows)}  β={len(beta_rows)}  Abs Rate={len(ai_rows)}")
        if not ai_rows:
            print(f"  [skip] {ds_name}: no Abstention Inflation samples in summary — P1/P3/P4 require Abs Rate.")
            return

        # Cap pool sizes to keep memory + runtime sane.
        gamma_rows = gamma_rows[: self.max_train_pool_each]
        beta_rows = beta_rows[: self.max_train_pool_each]
        if self.max_ai_samples > 0:
            ai_rows = ai_rows[: self.max_ai_samples]

        # α-pool from supplementary FLD summary.
        alpha_rows = self._read_alpha_pool()
        alpha_rows = alpha_rows[: self.max_train_pool_each]
        print(f"  α-pool from supp FLD: {len(alpha_rows)}")

        # Resolve sample_id → Sample object so we can rebuild prompts
        # (deterministic — same builders the AB runner used).
        samples_all = self.data_handler.load_dataset(ds_name)
        samples_all = [s for s in samples_all if s.answer_idx >= 0]
        id_to_sample = {s.id: s for s in samples_all}

        # Build the prompt sets we will need to forward through the local model.
        ai_samples = [id_to_sample[r["id"]] for r in ai_rows if r["id"] in id_to_sample]
        gamma_samples = [id_to_sample[r["id"]] for r in gamma_rows if r["id"] in id_to_sample]
        beta_samples = [id_to_sample[r["id"]] for r in beta_rows if r["id"] in id_to_sample]
        alpha_samples = self._load_alpha_samples(alpha_rows)

        ai_s1_prompts = [self._s1_prompt(s) for s in ai_samples]
        ai_s2_prompts = [self._s2_prompt(s) for s in ai_samples]
        gamma_s1_prompts = [self._s1_prompt(s) for s in gamma_samples]
        beta_s1_prompts = [self._s1_prompt(s) for s in beta_samples]
        alpha_s2_prompts = [self._s2_prompt(s) for s in alpha_samples]

        # Lazy model load (gives us a clean error before paying tokenization cost).
        self._ensure_model()

        # P1 — extract S1 + S2 hidden states on Abstention Inflation samples; compute AID.
        print("  [P1] extracting Abs Rate S1 / S2 hidden states ...")
        ai_s1_hs = self._extract(ai_s1_prompts)
        ai_s2_hs = self._extract(ai_s2_prompts)
        p1 = fit_p1(ai_s1_hs, ai_s2_hs)
        print(f"  [P1] best layer ℓ*={p1.best_layer}  AUC={p1.per_layer_auc[p1.best_layer]:.3f}  "
              f"random_baseline={p1.random_baseline_auc:.3f}")

        # P2 — letter probe trained on γ-pool S1 hidden states; eval on Abs Rate S2.
        print("  [P2] training letter probe on γ pool ...")
        gamma_s1_hs = self._extract(gamma_s1_prompts) if gamma_samples else np.zeros((0, *ai_s1_hs.shape[1:]))
        gamma_letters = [_letter_for(r) for r in gamma_rows[: len(gamma_samples)]]
        ai_gold_letters = [_letter_for(r) for r in ai_rows[: len(ai_samples)]]
        p2 = fit_p2(
            gamma_s1_hs, gamma_letters,
            ai_s2_hs, ai_gold_letters,
            layer=p1.best_layer,
        )
        print(f"  [P2] train_acc={p2.train_acc:.3f}  test_acc_on_ai={p2.test_acc_on_ai:.3f}")

        # P3 — 3-way attribution at ℓ*. Need β + α hidden states.
        print("  [P3] extracting β / α hidden states ...")
        beta_s1_hs = self._extract(beta_s1_prompts) if beta_samples else np.zeros((0, *ai_s1_hs.shape[1:]))
        alpha_s2_hs = self._extract(alpha_s2_prompts) if alpha_samples else np.zeros((0, *ai_s1_hs.shape[1:]))
        p3 = fit_p3(
            gamma_s1_hs[:, p1.best_layer, :] if gamma_s1_hs.shape[0] else np.zeros((0, ai_s1_hs.shape[2])),
            beta_s1_hs[:, p1.best_layer, :] if beta_s1_hs.shape[0] else np.zeros((0, ai_s1_hs.shape[2])),
            alpha_s2_hs[:, p1.best_layer, :] if alpha_s2_hs.shape[0] else np.zeros((0, ai_s1_hs.shape[2])),
            ai_hs=ai_s2_hs[:, p1.best_layer, :],
            layer=p1.best_layer,
        )
        print(f"  [P3] buckets={p3.bucket_counts}  high_conf_share={p3.high_conf_share:.2%}")

        # P4 — directional ablation, by-bucket recovery.
        p4_dict: Optional[Dict[str, Any]] = None
        if self.run_p4_flag and p3.ai_labels:
            print("  [P4] directional ablation (this is slow — one generation per Abstention Inflation sample) ...")
            p4 = run_p4(
                self._model, self._tokenizer, self._device,
                s2_prompts=ai_s2_prompts,
                gold_letters=ai_gold_letters,
                p3_labels=p3.ai_labels,
                layer=p1.best_layer,
                direction=p1.aid_unit_at_best,
                max_new_tokens=self.p4_max_new_tokens,
            )
            print(f"  [P4] mech_recovery={p4.mech_recovery:.3f}  "
                  f"correct_recovery={p4.correct_recovery:.3f}")
            p4_dict = self._p4_to_dict(p4)
        else:
            print("  [P4] skipped (run_p4=false or no Abs Rate labels).")

        # ------------------------------------------------------------------
        # Persist
        # ------------------------------------------------------------------
        out_path = self.results_dir / f"probe_1d_{ds_name}_{self.probe_label}.json"
        out = {
            "dataset": ds_name,
            "summary_model": self.summary_model_name,
            "probe_model_path": self.local_model_path,
            "probe_label": self.probe_label,
            "n_pools": {
                "gamma": len(gamma_samples),
                "beta":  len(beta_samples),
                "alpha": len(alpha_samples),
                "abs_rate":   len(ai_samples),
            },
            "p1": self._p1_to_dict(p1),
            "p2": self._p2_to_dict(p2),
            "p3": self._p3_to_dict(p3),
            "p4": p4_dict,
            "per_ai_sample": [
                {
                    "sample_id":   r["id"],
                    "answer_idx":  r["answer_idx"],
                    "p2_pred_letter": p2.ai_pred_letters[i] if i < len(p2.ai_pred_letters) else None,
                    "p2_confidence":  p2.ai_confidences[i]  if i < len(p2.ai_confidences)  else 0.0,
                    "p3_label":       p3.ai_labels[i]       if i < len(p3.ai_labels)       else None,
                    "p3_confidence":  p3.ai_confidences[i]  if i < len(p3.ai_confidences)  else 0.0,
                    "p3_probs":       p3.ai_probs[i].tolist() if i < p3.ai_probs.shape[0]  else None,
                    "p4_pred_letter": (p4_dict["pred_letters"][i] if p4_dict else None),
                }
                for i, r in enumerate(ai_rows[: len(ai_samples)])
            ],
        }
        self.data_handler.save_json(out, out_path)
        print(f"  [saved] {out_path}")

    # =================================================================
    # Helpers
    # =================================================================
    def _read_ab_summary(self, ds_name: str) -> Optional[dict]:
        path = self.ab_results_dir / f"ab_summary_{ds_name}_{self.summary_model_name}.json"
        if not path.exists():
            print(f"  [skip] {ds_name}: missing ab_summary at {path}")
            return None
        return json.loads(path.read_text())

    def _read_alpha_pool(self) -> List[dict]:
        path = self.supp_results_dir / f"supp_summary_FLD_{self.summary_model_name}.json"
        if not path.exists():
            print(f"  [warn] α-pool source not found: {path}; α-pool will be empty.")
            return []
        s = json.loads(path.read_text())
        # α: model correctly answered Unknown on a genuinely-Unknown FLD sample
        # under the S2 setting.
        return [r for r in s.get("per_sample", []) if r.get("pred_s2") == "UNKNOWN"]

    def _load_alpha_samples(self, alpha_rows: List[dict]):
        """Re-load FLD genuinely-Unknown samples that match alpha_rows."""
        from core.dataset_loader import load_judge
        all_fld = load_judge("FLD")
        unknown_only = [s for s in all_fld if s.answer_idx == -1]
        ids = {r["id"] for r in alpha_rows}
        return [s for s in unknown_only if s.id in ids]

    def _s1_prompt(self, sample):
        if sample.task_type == "mcq":
            return build_mcq_s1_prompt(sample.question, sample.options)
        return build_judge_s1_prompt(get_scheme(sample.source), sample.question, sample.context)

    def _s2_prompt(self, sample):
        if sample.task_type == "mcq":
            return build_mcq_s2_prompt(sample.question, sample.options)
        return build_judge_s2_prompt(get_scheme(sample.source), sample.question, sample.context)

    def _ensure_model(self):
        if self._model is not None:
            return
        print(f"  [load] {self.local_model_path} (dtype={self.dtype})")
        self._model, self._tokenizer, self._device = hs_mod.load_probe_model(
            self.local_model_path, device=self.device, dtype=self.dtype,
        )

    def _extract(self, prompts: List[List[dict]]) -> np.ndarray:
        if not prompts:
            return np.zeros((0,), dtype=np.float32)
        return hs_mod.extract_answer_token_hidden_states(
            self._model, self._tokenizer, self._device, prompts,
            batch_size=self.batch_size, max_length=self.max_seq_length,
        )

    # ------------------------------------------------------------------
    # Serialization helpers — keep the JSON small + numpy-free.
    # ------------------------------------------------------------------
    @staticmethod
    def _p1_to_dict(p1: P1Result) -> dict:
        return {
            "best_layer":          p1.best_layer,
            "best_layer_auc":      float(p1.per_layer_auc[p1.best_layer]),
            "random_baseline_auc": float(p1.random_baseline_auc),
            "per_layer_auc":       p1.per_layer_auc.tolist(),
        }

    @staticmethod
    def _p2_to_dict(p2: P2Result) -> dict:
        return {
            "layer":             p2.layer,
            "train_acc":         p2.train_acc,
            "test_acc_on_ai":   p2.test_acc_on_ai,
            "mean_ai_confidence": (
                float(np.mean(p2.ai_confidences)) if p2.ai_confidences else 0.0
            ),
        }

    @staticmethod
    def _p3_to_dict(p3: P3Result) -> dict:
        return {
            "layer":            p3.layer,
            "train_acc":        p3.train_acc,
            "bucket_counts":    p3.bucket_counts,
            "high_conf_share":  p3.high_conf_share,
            "pool_sizes":       p3.pool_sizes,
        }

    @staticmethod
    def _p4_to_dict(p4: P4Result) -> dict:
        return {
            "layer":             p4.layer,
            "mech_recovery":     p4.mech_recovery,
            "correct_recovery":  p4.correct_recovery,
            "by_p3_label":       p4.by_p3_label,
            "pred_letters":      p4.pred_letters,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_correct(letter: str, answer_idx: int) -> bool:
    if letter not in _LETTERS:
        return False
    return ord(letter) - ord("A") == answer_idx


def _letter_for(row: dict) -> str:
    """The gold letter for a per_sample row. answer_idx ∈ {0..3}."""
    ai = row["answer_idx"]
    if ai < 0 or ai >= len(_LETTERS):
        return "A"  # fallback; FLD γ-pool letters are validated separately
    return _LETTERS[ai]
