"""Unified dataset loader.

Every paper benchmark (FLD, FOLIO, ARC, MedQA, MMLU, LogiQA, plus the two
truly-Unknown subsets ``FLD_unknown`` / ``FOLIO_unknown``) is stored in
``dataset/<name>.json`` under a single canonical schema:

    {
      "id":          str,           # stable per-item identifier
      "question":    str,           # claim (TFQ) or question stem (MCQ)
      "context":     str,           # FLD Facts / FOLIO Premises / "" for MCQ
      "options":     list[str],     # ["True","False"] for TFQ; 4 strings for MCQ
      "answer":      str,           # gold answer text ("True"/"False"/"Unknown" or option text)
      "answer_idx":  int,           # 0..N-1, or -1 for truly-Unknown samples
      "task_type":   "tf" | "mcq",
      "source":      str,           # dataset name (FLD / FOLIO / ARC / ...)
      "depth":       int,           # FLD only: proof-tree step count (S10 difficulty)
    }

Each row converts directly to ``Sample`` via :py:meth:`Sample.from_dict`.

Routing rule
------------
``load_dataset(name)`` returns the full list of items in
``dataset/<name>.json``. Truly-Unknown subsets used by S9 truly-unknown
perception are addressed as ``FLD_unknown`` / ``FOLIO_unknown``. For S2-style
"answerable only" experiments, filter the returned list with
``[s for s in samples if s.answer_idx >= 0]``.

The on-disk schema is identical across datasets, so no per-dataset loader
or per-dataset label remapping is needed — everything else (prompt verbs,
context labels, task instructions, output parser) is owned by
:py:class:`infra.label_scheme.LabelScheme`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Canonical dataset directory bundled with this software package.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"

# Datasets enumerated in the paper (Section "Datasets"). Truly-Unknown subsets
# carry the ``_unknown`` suffix and supply gold answer_idx = -1.
PAPER_DATASETS = (
    "FLD", "FLD_unknown",
    "FOLIO", "FOLIO_unknown",
    "ARC", "MedQA", "MMLU", "LogiQA",
)


@dataclass
class Sample:
    """One evaluation item.

    Attributes match the on-disk schema; any extra fields not in the named
    set below (e.g. FLD's ``depth``) land in :pyattr:`extra` so analyses that
    need them (S10 difficulty) keep working without schema churn.
    """

    id: str
    question: str
    options: List[str]      # ["True","False"] for TFQ; 4 strings for MCQ
    answer_idx: int         # 0..N-1; -1 = truly-Unknown gold label
    task_type: str = "mcq"  # "tf" | "mcq"
    source: str = ""
    context: str = ""
    answer: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        known = {"id", "question", "options", "answer_idx",
                 "task_type", "source", "context", "answer"}
        return cls(
            id=str(d["id"]),
            question=d["question"],
            options=list(d["options"]),
            answer_idx=int(d["answer_idx"]),
            task_type=d.get("task_type", "mcq"),
            source=d.get("source", ""),
            context=d.get("context", ""),
            answer=d.get("answer", ""),
            extra={k: v for k, v in d.items() if k not in known},
        )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def load_dataset(name: str,
                 root: Optional[Path] = None) -> List[Sample]:
    """Return every item in ``<root>/<name>.json`` as a ``Sample`` list.

    The package ships with all eight paper datasets under ``software/dataset/``
    so the default invocation needs no path argument.

    Parameters
    ----------
    name : str
        Dataset stem, e.g. ``"FLD"``, ``"FOLIO_unknown"``, ``"MedQA"``.
    root : Path, optional
        Override the dataset directory; defaults to the bundled
        ``software/dataset/`` directory.
    """
    root = Path(root) if root is not None else DEFAULT_DATASET_ROOT
    path = root / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path}. Known paper datasets: "
            f"{', '.join(PAPER_DATASETS)}."
        )
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return [Sample.from_dict(d) for d in items]


# Pre-refactor aliases. The repo used to ship one loader per dataset family
# (``load_judge`` for FLD/FOLIO, ``load_arc`` / ``load_medqa`` for MCQ); they all
# took a dataset name and are now the same unified call.
load_judge = load_dataset
load_arc = load_dataset
load_medqa = load_dataset
load_mmlu = load_dataset
load_logiqa = load_dataset


def answerable_subset(samples: List[Sample]) -> List[Sample]:
    """Filter to items whose gold label is a committed True/False/option.

    The paper computes S2 Abs Rate over this subset; truly-Unknown items
    are evaluated separately by S9 (and live in the ``*_unknown`` files).
    """
    return [s for s in samples if s.answer_idx >= 0]


def truly_unknown_subset(samples: List[Sample]) -> List[Sample]:
    """Filter to gold ``Unknown`` items (answer_idx == -1)."""
    return [s for s in samples if s.answer_idx == -1]


def apply_sample_limit(samples: List[Sample], spec) -> List[Sample]:
    """Honour a ``sample_limits`` directive from a YAML config.

    Supported forms (mirroring the ones used in the paper's configs):

    * ``None``            — keep everything.
    * ``int``             — keep the first ``N`` items.
    * ``{"true": N1, "false": N2}`` — keep the first ``N1`` items with
      answer_idx == 0 (True) and the first ``N2`` items with answer_idx == 1
      (False). Useful for the balanced S1/S2 runs reported in the paper
      (e.g. 250 True + 250 False on FLD/FOLIO). Aliases ``proved`` /
      ``disproved`` are accepted for backward compatibility with older
      configs.
    """
    if spec is None:
        return samples
    if isinstance(spec, int):
        return samples[:spec]
    if isinstance(spec, dict):
        # Normalise key aliases: True ↔ 0, False ↔ 1.
        key_to_idx = {
            "true": 0, "True": 0, "TRUE": 0, "proved": 0, "PROVED": 0,
            "false": 1, "False": 1, "FALSE": 1, "disproved": 1, "DISPROVED": 1,
        }
        out: List[Sample] = []
        for raw_key, n in spec.items():
            if raw_key not in key_to_idx:
                raise ValueError(
                    f"Unrecognised sample_limits key {raw_key!r}; "
                    f"use 'true'/'false' (or 'proved'/'disproved')."
                )
            target_idx = key_to_idx[raw_key]
            picked = [s for s in samples if s.answer_idx == target_idx][:n]
            out.extend(picked)
        return out
    raise TypeError(
        f"sample_limits must be int | dict | None, got {type(spec).__name__}."
    )
