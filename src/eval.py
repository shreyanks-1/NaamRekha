"""seqeval entity-level micro-F1, overall and per type."""
# Module docstring: this file wraps the `seqeval` library to compute
# entity-level (not token-level) F1 scores for NER predictions — meaning a
# predicted entity only counts as correct if its full span and type match
# the gold span exactly, which is the standard NER evaluation convention.

from __future__ import annotations
# Enables postponed evaluation of annotations (PEP 563), so modern generic
# type hints like `list[list[str]]` work smoothly in the function signature below.

from typing import cast
# `cast` is a typing-only helper (no runtime effect) used to tell static type
# checkers "trust me, this value has this type" — used below because
# classification_report's return type isn't precisely typed upstream.

from seqeval.metrics import classification_report
# Import seqeval's classification_report, which computes precision/recall/F1
# per entity type and as a micro/macro average, specifically understanding
# BIO-tagged sequences (so it correctly judges whole entity spans).

# result key -> seqeval entity label
_TYPES = {"F1_PER": "PER", "F1_ORG": "ORG", "F1_LOC": "LOC"}
# Maps this project's desired output dict keys (e.g. "F1_PER") to the entity
# type name seqeval itself uses internally (e.g. "PER") after stripping the
# B-/I- BIO prefixes — lets us rename seqeval's keys to our result schema.


def evaluate(y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, float]:
    """Score predicted tags against gold (one tag sequence per sentence).
    Returns F1_overall, F1_PER, F1_ORG, F1_LOC."""
    # Docstring: documents the expected input shape (a list of sentences,
    # each a list of BIO tag strings) and exactly which four scores it returns.
    if len(y_true) != len(y_pred):
        # Sanity-check that gold and predicted sequences have the same number
        # of sentences — a mismatch here would silently misalign scoring.
        raise ValueError(
            f"y_true and y_pred differ in length: {len(y_true)} vs {len(y_pred)}"
            # Error message reports both lengths so the mismatch is immediately
            # diagnosable without needing to add print statements.
        )

    report = cast(dict, classification_report(y_true, y_pred, output_dict=True))
    # Run seqeval's evaluation: output_dict=True makes it return a nested
    # dict (instead of a printable string report) with per-label and
    # averaged precision/recall/F1/support; `cast(dict, ...)` just satisfies
    # the type checker since seqeval's stub may type this more loosely.
    result = {"F1_overall": float(report["micro avg"]["f1-score"])}
    # Start the result dict with the overall score: "micro avg" aggregates
    # true/false positives/negatives across all entity types before computing
    # F1, which (for NER) is the conventional "overall F1" metric; cast to
    # plain Python float since numpy/stdlib numeric types can vary.
    for col, entity in _TYPES.items():
        # Iterate each (result-column-name, seqeval-entity-label) pair, e.g.
        # ("F1_PER", "PER"), to build the per-entity-type scores.
        result[col] = float(report.get(entity, {}).get("f1-score", 0.0))
        # Look up this entity type's sub-report; if the entity type never
        # appeared in y_true/y_pred at all, report.get(entity, {}) safely
        # falls back to an empty dict, and the inner .get(..., 0.0) then
        # defaults the F1 score to 0.0 instead of raising a KeyError.
    return result
    # Return the combined dict: {"F1_overall": ..., "F1_PER": ..., "F1_ORG": ..., "F1_LOC": ...}.
