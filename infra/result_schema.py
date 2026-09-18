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
``pred_s4`` / ``S4``    forced rerun on abstaining samples  ``S5``  (paper S5)
======================  ==================================  ==================

``S1`` and ``S2`` are stable across both eras, so files written by the current
code keep those key names. The two ambiguous settings get *new, unambiguous*
key names (``*_s3_format``, ``*_s5_rerun``) instead of being silently
redefined, and new files carry ``"schema": SCHEMA_VERSION``.

Pre-rename files also carried a ``pred_s3`` holding a mitigation variant that
no paper setting uses. It is no longer translated, which also means S9's own
``pred_s3`` -- its third, calibration-suffix condition on Unknown-labeled
samples -- survives a read intact instead of being renamed out from under the
analysis.

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

Read every ``*.json`` through :func:`load_summary` and you get the
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
}

#: legacy setting name -> canonical setting name.
LEGACY_SETTING_MAP: Dict[str, str] = {
    "S1": "S1",
    "S2": "S2",
    # Legacy ``S3`` was a mitigation variant no paper setting uses. It still
    # needs a target: legacy ``S5`` becomes the paper's ``S3``, so leaving
    # legacy ``S3`` to fall through to itself puts two different settings on
    # one key, and whichever the file happens to list second wins -- silently,
    # and differently per file. Parking it on a dead name keeps the collision
    # impossible instead of merely unlikely.
    "S3": "legacy_mitigation_dropped",
    "S4": "S5",
    "S5": "S3",
}

#: legacy per-sample key -> canonical per-sample key.
LEGACY_SAMPLE_KEY_MAP: Dict[str, str] = {
    "pred_s1": "pred_s1",
    "raw_s1": "raw_s1",
    "pred_s2": "pred_s2",
    "raw_s2": "raw_s2",
    "pred_s4": "pred_s5_rerun",
    "raw_s4": "raw_s5_rerun",
    "pred_s5": "pred_s3_format",
    "raw_s5": "raw_s3_format",
}


#: canonical summary key -> pre-unification spellings, newest alias first.
#: Written by code that predates the AIR -> "Abs Rate" terminology unification.
LEGACY_FIELD_ALIASES: Dict[str, tuple] = {
    "n_abstention_inflation": ("n_air_s2", "n_air"),
    "s5_rerun": ("air_followups",),
    "abs_rate": ("air", "AIR"),
    "abs_rate_s2": ("AIR_S2", "air_s2"),
    "abs_rate_s3": ("AIR_S3", "air_s3"),
}

#: legacy ``sample_type`` value -> canonical value (S8 logit-lens records).
LEGACY_SAMPLE_TYPES: Dict[str, str] = {
    "AIR": "ai",
    "air": "ai",
    "non_AIR": "non_ai",
    "non_air": "non_ai",
    "nonAIR": "non_ai",
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
    out["metrics"] = {LEGACY_SETTING_MAP.get(k, k): v for k, v in metrics.items()}

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
            k: (
                {**v, "abs_rate": v["air"]}
                if isinstance(v, dict) and "air" in v and "abs_rate" not in v
                else v
            )
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


#: Every setting the paper reports, and the folder under results/ its files
#: live in. One registry: the runners write where it says, the analyses read
#: where it says, and :func:`stamp` puts the same key into every file so a
#: file names its own setting without its folder. A variant separates the two
#: halves a setting has (S4's synonyms and random words, S9's two probes,
#: S10's temperature and size sweeps); "{slug}" is the model slug where one
#: sweep is stored per model.
SETTING_DIRS = {
    "S1": "S1_baseline",
    "S2": "S2_unknown_option",
    "S3": "S3_question_format",
    "S4/synonyms": "S4_word_content/synonyms",
    "S4/random_words": "S4_word_content/random_words",
    "S5": "S5_rerun",
    "S6": "S6_self_diagnosis",
    "S7": "S7_reasoning_traces",
    "S8": "S8_logit_lens",
    "S9/Perception_Unknown_labeled_Samples": "S9_stability/Perception_Unknown_labeled_Samples",
    "S9/Persistence_Across_Repeats": "S9_stability/Persistence_Across_Repeats",
    "S10/temperature": "S10_factor_analysis/temperature",  # one subfolder per model slug
    "S10/size_alignment": "S10_factor_analysis/size_alignment",
    "S11": "S11_positional_bias",
}


def results_dir(key: str, root: str | Path = "results", **fmt) -> Path:
    """The folder for ``key`` (``"S6"``, ``"S4/synonyms"``, ...) under ``root``."""
    return Path(root) / SETTING_DIRS[key].format(**fmt)


def stamp(key: str) -> Dict[str, str]:
    """The fields every result file carries so it names its own setting.

    ``{"setting": "S4", "variant": "synonyms"}`` for ``"S4/synonyms"``;
    ``{"setting": "S6"}`` when the setting has no variant.
    """
    setting, _, variant = key.partition("/")
    return {"setting": setting, **({"variant": variant} if variant else {})}


def setting_key_for_dir(rel_dir: str) -> str | None:
    """Registry key for a directory given relative to results/ (e.g.
    ``"S4_word_content/synonyms"`` or ``"S1_baseline/tfq/gpt_5.4_nano"``); the inverse
    of :data:`SETTING_DIRS`, for files written before they were stamped."""
    rel = str(rel_dir).strip("/")
    best = None
    for key, pat in SETTING_DIRS.items():
        if rel == pat or rel.startswith(pat + "/"):
            if best is None or len(pat) > len(SETTING_DIRS[best]):
                best = key
    return best


_PAIRED_FIELD = {
    "S1": ("pred_s1", "raw_s1"),
    "S2": ("pred_s2", "raw_s2"),
    "S3": ("pred_s3_format", "raw_s3_format"),
}


def iter_cells(root: str | Path = "results"):
    """Every main-experiment cell as ``(dataset, model, slug, task_type)``.

    Enumerated from S1, which every cell has; the other settings are joined
    onto it by :func:`load_cell`.
    """
    base = Path(root) / SETTING_DIRS["S1"]
    for family, task_type in (("tfq", "tf"), ("mcq", "mcq")):
        for f in sorted((base / family).glob("*/*.json")):
            dataset, _, model = f.stem.partition("_")
            yield dataset, model, f.parent.name, task_type


#: S11 puts the "Unknown" option in one of three slots; files are named by the
#: slot's position in the option list, as the paper's figure labels them.
POSITION_NAME = {"A": "first", "B": "second", "C": "last"}


def position_name(slot: str) -> str:
    """``"A"`` -> ``"first"``, ``"B"`` -> ``"second"``, ``"C"`` -> ``"last"``."""
    return POSITION_NAME[slot]


#: model name as the endpoint reports it -> the folder its cells live in. The
#: slug is the name written out with the separators the file system prefers.
MODEL_SLUG = {
    "Olmo-3-7B-Instruct": "olmo_3_7b_instruct",
    "gpt-5.4-nano": "gpt_5.4_nano",
    "deepseek-v4-flash": "deepseek_v4_flash",
    "gemini-3.1-flash-lite": "gemini_3.1_flash_lite",
}


def model_slug(model_name: str) -> str:
    """Folder name for a model; falls back to the name with '-' -> '_'."""
    return MODEL_SLUG.get(model_name, model_name.replace("/", "_").replace("-", "_"))


#: S8 probes the Olmo-3-7B family on the TFQ datasets: FLD is the paper's run
#: and the default of every S8 path, FOLIO the same chain on the other TFQ
#: dataset. One table is the source of every S8
#: name: the code's checkpoint key -> the HuggingFace repo it was downloaded
#: from. The download folder, the path the probe loads and the result file
#: are all named after the repo's own basename, so a file says exactly which
#: checkpoint produced it.
S8_MODEL = "olmo-3-7b"  # the family; names the results subfolder
S8_CHECKPOINTS = {
    "base": "allenai/Olmo-3-1025-7B",
    "sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "instruct": "allenai/Olmo-3-7B-Instruct",
    "rl_zero": "allenai/Olmo-3-7B-RL-Zero-General",
}
#: The checkpoint whose S1/S2 answers partition a dataset's items into abstention
#: inflation / correct abstention; every probe is scored on that partition.
S8_INFERENCE_CKPT = "instruct"
#: The datasets the S8 chain runs on, and the one every S8 path defaults to.
S8_DATASETS = ("FLD", "FOLIO")
S8_DEFAULT_DATASET = "FLD"


def s8_checkpoint_name(ckpt: str) -> str:
    """``"instruct"`` -> ``"Olmo-3-7B-Instruct"``, the repo's own basename."""
    return S8_CHECKPOINTS[ckpt].rsplit("/", 1)[-1]


def s8_model_dir(ckpt: str, models_root: str | Path = "models") -> Path:
    """Where step 1 downloads a checkpoint and steps 2-3 load it from."""
    return Path(models_root) / s8_checkpoint_name(ckpt)


def s8_inference_path(
    root: str | Path = "results", dataset: str = S8_DEFAULT_DATASET
) -> Path:
    """Step 2's S1/S2 inference, the intermediate the probes are built from.

    Named for the dataset and the checkpoint that produced it. Not published:
    each probe row carries the pred_s1 / pred_s2 it needs, so results/ holds
    only the probes."""
    return (
        results_dir("S8", root)
        / model_slug(S8_MODEL)
        / f"{dataset}_{s8_checkpoint_name(S8_INFERENCE_CKPT)}_inference.json"
    )


def s8_logit_lens_path(
    ckpt: str, root: str | Path = "results", dataset: str = S8_DEFAULT_DATASET
) -> Path:
    """The logit-lens probe for one checkpoint: olmo_3_7b/<dataset>_<checkpoint name>.json."""
    return (
        results_dir("S8", root)
        / model_slug(S8_MODEL)
        / f"{dataset}_{s8_checkpoint_name(ckpt)}.json"
    )


#: Settings collected for TFQ only; their folders skip the {tfq,mcq} level.
TFQ_ONLY = {"S3"}


def model_dir(key: str, model: str, root: str | Path = "results") -> Path:
    """``results_dir(key) / model_slug(model)`` -- where one model's cells of a
    setting live. Every setting that is split by model uses this."""
    d = results_dir(key, root) / model_slug(model)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cell_path(
    setting: str,
    dataset: str,
    model: str,
    slug: str,
    task_type: str = "tf",
    root: str | Path = "results",
) -> Path:
    """Where one (setting, dataset, model) cell lives."""
    base = Path(root) / SETTING_DIRS[setting]
    if setting not in TFQ_ONLY:  # S3 has no MCQ arm, so no family level
        base = base / ("tfq" if task_type == "tf" else "mcq")
    return base / slug / f"{dataset}_{model}.json"


def load_cell(
    dataset: str,
    model: str,
    slug: str,
    task_type: str = "tf",
    root: str | Path = "results",
) -> Dict[str, Any]:
    """Join a cell's settings back into one paired summary.

    Reads whichever of S1/S2/S3/S5 exist for this cell and returns them in the
    shape the analyses expect: ``per_sample`` rows carrying ``pred_s1`` /
    ``pred_s2`` / ``pred_s3_format`` for the same item, plus ``s5_rerun``
    alongside. The join is by item id, so a setting that is missing an item
    simply leaves its field unset and :func:`paired_keep_ids` drops the item,
    exactly as it did when the four settings shared one file.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    metrics, ran, head = {}, [], {}
    for setting in ("S1", "S2", "S3"):
        path = cell_path(setting, dataset, model, slug, task_type, root)
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        ran.append(setting)
        metrics[setting] = doc.get("metrics", {})
        head = head or doc
        pf, rf = _PAIRED_FIELD[setting]
        for r in doc["per_sample"]:
            row = rows.setdefault(
                r["id"],
                {
                    "id": r["id"],
                    "source": r.get("source"),
                    "answer_idx": r["answer_idx"],
                },
            )
            row[pf], row[rf] = r["pred"], r.get("raw")
    # An item one setting never returned must leave the paired contrast, as it
    # did when the settings shared a file: mark it unparseable with a raw that
    # classifies as a non-reply, and paired_keep_ids drops it.
    for row in rows.values():
        for setting in ran:
            pf, rf = _PAIRED_FIELD[setting]
            if pf not in row:
                row[pf], row[rf] = "UNPARSEABLE", "__MISSING_IN_" + setting + "__"
    out: Dict[str, Any] = {
        "schema": head.get("schema", SCHEMA_VERSION),
        "dataset": dataset,
        "model": model,
        "task_type": head.get("task_type", task_type),
        "trace_family": head.get("trace_family"),
        "settings_run": ran,
        "n_total": len(rows),
        "metrics": metrics,
        "per_sample": list(rows.values()),
    }
    s5 = cell_path("S5", dataset, model, slug, task_type, root)
    if s5.exists():
        doc = json.loads(s5.read_text(encoding="utf-8"))
        out["settings_run"] = ran + ["S5"]
        out["metrics"]["S5"] = doc.get("metrics", {})
        out["s5_rerun"] = [
            {
                "sample_id": r["id"],
                "answer_idx": r["answer_idx"],
                "pred_s5_rerun": r["pred"],
                "raw_s5_rerun": r.get("raw"),
            }
            for r in doc["per_sample"]
        ]
        out["n_abstention_inflation"] = len(out["s5_rerun"])
    return out


def is_paired_summary(path: str | Path) -> bool:
    """True when ``path`` is a paired S1/S2 summary in the canonical schema.

    Decided by the ``schema`` field the writer stamps into the file, not by the
    file name: the name says which cell a file holds, the schema says how to
    read it, and only the second is a contract. Reads the head of the file so a
    30 MB summary costs the same as a small one.
    """
    path = Path(path)
    if path.suffix != ".json":
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(4096)
    except OSError:
        return False
    if '"setting"' in head:  # a per-setting cell, not the paired view
        return False
    if f'"{SCHEMA_VERSION}"' in head:
        return True
    if '"schema"' in head:  # some other schema, decided
        return False
    try:  # schema past the head, or absent
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        return (
            isinstance(doc, dict)
            and doc.get("schema") == SCHEMA_VERSION
            and "setting" not in doc
        )
    except (OSError, ValueError):
        return False


def find_paired_summaries(root: str | Path):
    """Every paired S1/S2 summary under ``root``, in sorted path order."""
    return [p for p in sorted(Path(root).rglob("*.json")) if is_paired_summary(p)]


def load_summary(path: str | Path) -> Dict[str, Any]:
    """Load a paired S1/S2 summary and normalize it to the canonical namespace."""
    with Path(path).open("r", encoding="utf-8") as f:
        return normalize(json.load(f))


def paired_keep_ids(
    summary: Dict[str, Any], settings: tuple[str, ...] = ("s1", "s2")
) -> set[str]:
    """Ids of the items an S1/S2 contrast is scored on.

    The runner drops an item from the paired contrast when a setting returned
    nothing usable -- an API error, a persistent failure, a content-filter
    refusal or a decoding collapse -- and keeps it when the model answered but
    declined to commit, which is a real response. Any analysis that reports a
    rate alongside the paper's accuracies has to use the same set, or it is
    quoting two different samples of the dataset as though they were one.

    This is the same rule ``ABRunner`` applies when it writes ``n_scored``;
    it lives here so downstream scripts do not each re-derive it.
    """
    from infra.evaluator import Evaluator

    ev = Evaluator()
    keep = set()
    for row in summary.get("per_sample", []):
        for name in settings:
            if row.get(f"pred_{name}") != "UNPARSEABLE":
                continue
            if ev.classify_unanswered(row.get(f"raw_{name}") or "") != "no_commitment":
                break
        else:
            keep.add(row["id"])
    return keep
