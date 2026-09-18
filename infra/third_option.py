"""Which setting a run belongs to, decided by the word in the third slot.

Every prompt in this study offers the model an extra option beyond the task's
own labels. Which word fills that slot is what separates the settings:

    "Unknown"                           S2, the main experiment
    a synonym of it                     S4, the word-content ablation
    an unrelated word                   S4, the random-word control

The three were routed by hand before, and the classes drifted apart: the
random-word control collected its own ``Unknown`` pass and stored it beside the
substituted words, so the two halves of S4 ended up measuring their shifts
against two different baselines. Deciding the destination from the word itself
keeps that from happening again -- an ``Unknown`` pass cannot be written into an
S4 directory, because :func:`results_dir` will not name one for it.
"""
from __future__ import annotations

from pathlib import Path

UNKNOWN = "Unknown"
#: Words that mean the same thing as "Unknown".
SYNONYMS = ("I don't know", "Indeterminate")
#: Words with no bearing on the question, used to test whether the *slot*
#: rather than the *word* is what triggers abstention.
RANDOM_WORDS = ("Triangular", "Cerulean")

#: class -> the directory that class's results live in. The main experiment is
#: split by task type and further by model slug, so it is named by family here
#: and completed from the config.
_DIRS = {
    "synonym": Path("results/S4_synonyms"),
    "random_word": Path("results/S4_random_words"),
}
_MAIN = {"tf": Path("results/S1_S3_tfq"), "mcq": Path("results/S1_S2_mcq")}


def slug(word: str) -> str:
    """File-name form of a third-option word: ``"I don't know"`` -> ``i_dont_know``.

    Apostrophes are dropped rather than turned into separators, so the slug
    reads as the contraction does.
    """
    kept = "".join(c for c in word.lower() if c.isalnum() or c.isspace())
    return "_".join(kept.split())


def classify(word: str) -> str:
    """``"unknown"``, ``"synonym"`` or ``"random_word"``."""
    if word == UNKNOWN:
        return "unknown"
    if word in SYNONYMS:
        return "synonym"
    if word in RANDOM_WORDS:
        return "random_word"
    raise ValueError(
        f"{word!r} is not a third-option word this study defines. Add it to "
        f"SYNONYMS or RANDOM_WORDS in infra/third_option.py and say which it is."
    )


def setting(word: str) -> str:
    """The setting a run with this third option belongs to: ``"S2"`` or ``"S4"``."""
    return "S2" if classify(word) == "unknown" else "S4"


def results_dir(word: str, task_type: str = "tf") -> Path:
    """Where a run with this third option writes.

    ``Unknown`` resolves to the main experiment's family directory; the model
    slug under it comes from the config, because that is where the main runs
    are addressed from. The two S4 classes resolve to their own directory.
    """
    kind = classify(word)
    if kind == "unknown":
        try:
            return _MAIN[task_type]
        except KeyError:
            raise ValueError(f"task_type must be 'tf' or 'mcq', got {task_type!r}")
    return _DIRS[kind]


def result_path(word: str, dataset: str, model: str, task_type: str = "tf") -> Path:
    """Full path for one cell. Refuses to name an S4 path for ``Unknown``.

    The baseline every S4 shift is measured against is the S2 cell of the main
    table, so there is no S4 file for ``Unknown`` to write.
    """
    if classify(word) == "unknown":
        raise ValueError(
            "Unknown is the S2 setting; its cell is the main experiment's, and "
            "both halves of S4 read it from there as their baseline. Point the "
            "run at the main experiment instead of writing a second copy."
        )
    safe_model = model.replace("/", "_")
    return results_dir(word, task_type) / f"{dataset}_{safe_model}_{slug(word)}.json"
