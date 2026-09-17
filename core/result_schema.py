"""Canonical result-file schema + reader for pre-rename summaries.

Why this module exists
----------------------
The experiments were run before the settings were renamed to the paper's
S1–S10. Two of the on-disk keys therefore mean something different in files
written before and after the rename:

======================  ==================================  ==================
legacy key              what it actually holds              canonical name
======================  ==================================  ==================
``pred_s1`` / ``S1``    baseline, no extra option           ``S1``
``pred_s2`` / ``S2``    "Unknown" option added              ``S2``
``pred_s5`` / ``S5``    TFQ re-rendered MCQ-style           ``S3``  (paper S3)
``pred_s3`` / ``S3``    S2 + calibration suffix             ``calibration_suffix``
``pred_s4`` / ``S4``    forced rerun on abstaining samples  ``S5``  (paper S5)
======================  ==================================  ==================

``S1`` and ``S2`` are stable across both eras, so files written by the current
code keep those key names. The two ambiguous settings get *new, unambiguous*
key names (``*_s3_format``, ``*_calibration_suffix``, ``*_s5_rerun``) instead of
being silently redefined, and new files carry ``"schema": SCHEMA_VERSION``.

A second rename followed, unifying the code's vocabulary with the paper's
(arXiv:2507.16199). The code had used ``AIR`` for two different things; the
paper has a separate term for each:

=========================  ==============================  ==================
legacy name                what it holds                   paper term
=========================  ==============================  ==================
``AIR`` / ``air``          fraction of items answered      **Abs Rate**
                           ``Unknown``                     (``abs_rate``)
``n_air_s2``               count of those items            ``n_abstention_inflation``
``air_followups``          the S5 rerun rows on them       ``s5_rerun``
``sample_type: "AIR"``     one such item                   **Abstention Inflation
                                                           sample** (``"ai"``)
``sample_type: "non_AIR"`` an item that did not abstain    ``"non_ai"``
=========================  ==============================  ==================

Read every ``ab_summary_*.json`` through :func:`load_summary` and you get the
canonical namespace regardless of when the file was written. For files read
with a bare ``json.load``, :func:`get_field` and :func:`canonical_sample_type`
resolve the legacy spellings individually.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

#: Provenance tiers whose label is an explicit commitment by the model, as
#: opposed to something a fallback scan inferred. `trusted_share` and
#: `abs_rate_strict` are both computed over these, so the tuple lives here
#: rather than as a literal in each producer -- it was duplicated in the S10
#: runner and the pass merger, and only one of the two was updated when
#: `final_line_runon` was added.
#:
#: * ``strict_em``        the whole response is the verb
#: * ``final_line``       the model's `Final answer:` line
#: * ``final_line_runon`` the same line, verb welded to trailing junk by a
#:                        detokenisation artifact (see `_runon_verb`)
TRUSTED_TIERS = ("strict_em", "final_line", "final_line_runon")

SCHEMA_VERSION = "paper-s1-s10/v1"

#: canonical setting name -> (per-sample pred key, per-sample raw key) written
#: by the current code.
CANONICAL_SAMPLE_KEYS: Dict[str, tuple] = {
    "S1": ("pred_s1", "raw_s1"),
    "S2": ("pred_s2", "raw_s2"),
    "S3": ("pred_s3_format", "raw_s3_format"),
    "S5": ("pred_s5_rerun", "raw_s5_rerun"),
    "calibration_suffix": ("pred_calibration_suffix", "raw_calibration_suffix"),
}

#: legacy setting name -> canonical setting name.
LEGACY_SETTING_MAP: Dict[str, str] = {
    "S1": "S1",
    "S2": "S2",
    "S3": "calibration_suffix",
    "S4": "S5",
    "S5": "S3",
}

#: legacy per-sample key -> canonical per-sample key.
LEGACY_SAMPLE_KEY_MAP: Dict[str, str] = {
    "pred_s1": "pred_s1", "raw_s1": "raw_s1",
    "pred_s2": "pred_s2", "raw_s2": "raw_s2",
    "pred_s3": "pred_calibration_suffix", "raw_s3": "raw_calibration_suffix",
    "pred_s4": "pred_s5_rerun", "raw_s4": "raw_s5_rerun",
    "pred_s5": "pred_s3_format", "raw_s5": "raw_s3_format",
}


#: canonical summary key -> pre-unification spellings, newest alias first.
#: Written by code that predates the AIR -> "Abs Rate" terminology unification.
LEGACY_FIELD_ALIASES: Dict[str, tuple] = {
    "n_abstention_inflation": ("n_air_s2", "n_air"),
    "s5_rerun":               ("air_followups",),
    "abs_rate":               ("air", "AIR"),
    "abs_rate_s2":            ("AIR_S2", "air_s2"),
    "abs_rate_s3":            ("AIR_S3", "air_s3"),
}

#: legacy ``sample_type`` value -> canonical value (S8 logit-lens records).
LEGACY_SAMPLE_TYPES: Dict[str, str] = {
    "AIR": "ai", "air": "ai",
    "non_AIR": "non_ai", "non_air": "non_ai", "nonAIR": "non_ai",
}


def get_field(obj: Dict[str, Any], canonical: str, default: Any = None) -> Any:
    """Read ``canonical`` from ``obj``, falling back to pre-rename spellings.

    Use this instead of a bare ``obj[key]`` wherever a result file may have
    been written before the terminology unification -- otherwise the new key
    is simply absent and the caller silently reads ``default`` (e.g. an Abs
    Rate of 0) from a perfectly good run.
    """
    if canonical in obj:
        return obj[canonical]
    for alias in LEGACY_FIELD_ALIASES.get(canonical, ()):
        if alias in obj:
            return obj[alias]
    return default


def canonical_sample_type(value: str) -> str:
    """Map a legacy ``sample_type`` value onto the canonical one (idempotent)."""
    return LEGACY_SAMPLE_TYPES.get(value, value)



def is_legacy(summary: Dict[str, Any]) -> bool:
    """True when the summary predates the S1–S10 rename."""
    return summary.get("schema") != SCHEMA_VERSION


def normalize(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``summary`` with legacy keys renamed to the canonical namespace.

    Idempotent: a summary already carrying ``SCHEMA_VERSION`` is returned
    unchanged. The input dict is not mutated.
    """
    if not is_legacy(summary):
        return summary

    out = dict(summary)
    out["schema"] = SCHEMA_VERSION
    out["schema_migrated_from"] = "legacy"

    metrics = summary.get("metrics") or {}
    out["metrics"] = {
        LEGACY_SETTING_MAP.get(k, k): v for k, v in metrics.items()
    }

    def _remap_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {LEGACY_SAMPLE_KEY_MAP.get(k, k): v for k, v in row.items()}

    if isinstance(summary.get("per_sample"), list):
        out["per_sample"] = [_remap_row(r) for r in summary["per_sample"]]

    # ``air_followups`` holds the S5 rerun rows in legacy files.
    if isinstance(summary.get("air_followups"), list):
        out["s5_rerun"] = [_remap_row(r) for r in summary["air_followups"]]
        out.pop("air_followups", None)

    n_ai = get_field(summary, "n_abstention_inflation")
    if n_ai is not None:
        out["n_abstention_inflation"] = n_ai
        out.pop("n_air_s2", None)
        out.pop("n_air", None)

    # Metric blocks: ``air`` / ``AIR_S2`` were the pre-unification names for
    # the paper's Abs Rate.
    if isinstance(out.get("metrics"), dict):
        out["metrics"] = {
            k: ({**v, "abs_rate": v["air"]} if isinstance(v, dict) and "air" in v
                and "abs_rate" not in v else v)
            for k, v in out["metrics"].items()
        }

    if isinstance(summary.get("tier_counts"), dict):
        out["tier_counts"] = {
            LEGACY_SETTING_MAP.get(k.upper(), k.upper()).lower(): v
            for k, v in summary["tier_counts"].items()
        }
    if isinstance(summary.get("n_unparseable"), dict):
        out["n_unparseable"] = {
            LEGACY_SETTING_MAP.get(k.upper(), k.upper()).lower(): v
            for k, v in summary["n_unparseable"].items()
        }
    return out


def load_summary(path: str | Path) -> Dict[str, Any]:
    """Load an ``ab_summary_*.json`` and normalize it to the canonical namespace."""
    with Path(path).open("r", encoding="utf-8") as f:
        return normalize(json.load(f))
