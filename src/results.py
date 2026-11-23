"""Append one run per call to results/results.csv with the fixed columns."""
# Module docstring: this file is the single place responsible for persisting
# experiment results (one row per training/evaluation run) to a shared CSV
# file, so every phase's notebook/script logs results in the same format.

from __future__ import annotations
# Enables postponed evaluation of type hints (PEP 563) for the modern
# `dict[str, float]` style annotations used in this file's function signature.

import csv
# Standard library csv module: used to write rows to the results file in
# proper CSV format (handling quoting/escaping automatically).
import os
# Standard library os module: used for path manipulation, directory
# creation, and checking whether the results file already exists.

COLUMNS = [
    # The fixed, ordered set of column names every results row must have —
    # defining this once ensures every run is logged with a consistent schema.
    ## WHY a fixed, explicit list instead of letting csv.DictWriter infer
    ## fieldnames from each row's dict keys: log_result is called from many
    ## separate scripts/notebooks (phase1 through phase6) over the life of
    ## the project, possibly weeks apart. If fieldnames were inferred per
    ## call, a row dict built in a slightly different key order, or one that
    ## happened to omit an optional field, would shift which column is
    ## which — and because results.csv is appended to, NOT rewritten, two
    ## different column orders could end up in the same file with only one
    ## header row at the top, making every row after the first drift
    ## written under the wrong column. The fixed COLUMNS list guarantees
    ## every row, from every phase/run, lines up under the same header.
    "model",
    # Which model architecture/checkpoint produced this run (e.g. "bilstm-crf", "muril").
    "train_lang",
    # The language code the model was trained on (e.g. "hi" for Hindi).
    "test_lang",
    # The language code the model was evaluated on (may differ from train_lang
    # in transfer-learning experiments).
    "subset_size",
    # How many training examples were used (useful for scaling-curve experiments).
    "epochs",
    # Number of training epochs the run used.
    "F1_PER",
    # Entity-level F1 score specifically for PERSON entities.
    "F1_ORG",
    # Entity-level F1 score specifically for ORGANIZATION entities.
    "F1_LOC",
    # Entity-level F1 score specifically for LOCATION entities.
    "F1_overall",
    # The micro-averaged F1 score across all entity types combined.
    "checkpoint_path",
    # Filesystem path to the saved model checkpoint for this run, if any.
    "notes",
    # Free-text field for any extra context about the run (e.g. hyperparameters).
]

# repo root is the parent of src/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# __file__ is this file's own path; abspath() makes it absolute, the first
# dirname() strips the filename to get the `src/` directory, and the second
# dirname() goes one level up from `src/` to the repository root — so this
# works regardless of the current working directory the script is run from.
RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "results.csv")
# The default location where results get appended: <repo_root>/results/results.csv.


def log_result(
    *,
    # The bare `*` forces every parameter after it to be passed by keyword
    # only (not positionally), preventing accidental argument-order mistakes
    # when calling this function from notebooks.
    ## WHY this matters concretely here: train_lang, test_lang are both plain
    ## `str` with no type distinction between them, and model/notes are also
    ## both `str`. Without keyword-only enforcement, a call like
    ## log_result("muril", "hi", "mr", 1000, 5, scores) would TYPE-CHECK fine
    ## even if someone swapped train_lang and test_lang by accident — Python
    ## can't catch a same-type argument-order mistake. The row would be
    ## logged with train/test languages reversed with no error, silently
    ## corrupting the scaling-curve/transfer-matrix analysis built on top of
    ## results.csv. Keyword-only args force every call site to write
    ## `train_lang=..., test_lang=...` explicitly, making the mistake visible
    ## at the call site instead of only in the downstream data.
    model: str,
    # Name/identifier of the model architecture used in this run.
    train_lang: str,
    # Language the model was trained on.
    test_lang: str,
    # Language the model was evaluated on.
    subset_size: int,
    # Size of the training subset used for this run.
    epochs: int,
    # Number of training epochs for this run.
    scores: dict[str, float],
    # The dict returned by eval.evaluate(), containing F1_overall/F1_PER/F1_ORG/F1_LOC.
    checkpoint_path: str = "",
    # Optional path to the saved model weights; defaults to empty string if not provided.
    notes: str = "",
    # Optional free-text notes about the run; defaults to empty string.
    path: str = RESULTS_PATH,
    # Which CSV file to append to; defaults to the shared project-wide results file,
    # but can be overridden (e.g. for a per-phase results file).
) -> None:
    """Append one run. scores is the dict from eval.evaluate.
    Writes the header if the file is new or empty."""
    # Docstring: documents that `scores` must come from eval.evaluate's
    # return shape, and that the CSV header is auto-written on first use.
    row = {
        # Build the dict for exactly one CSV row, mapping each COLUMNS name
        # to its corresponding value for this run.
        "model": model,
        # Store the model name as-is in this row.
        "train_lang": train_lang,
        # Store the training language code.
        "test_lang": test_lang,
        # Store the evaluation language code.
        "subset_size": subset_size,
        # Store the training subset size.
        "epochs": epochs,
        # Store the epoch count.
        "F1_PER": scores.get("F1_PER", ""),
        # Pull the PERSON F1 score out of the scores dict; falls back to an
        # empty string if that key is somehow missing, rather than raising.
        ## WHY .get(..., "") instead of scores["F1_PER"] here (and for the
        ## three F1 keys below): eval.evaluate() always returns all four
        ## keys today, but log_result's `scores` parameter is typed as the
        ## general `dict[str, float]`, not tied by the type system to
        ## evaluate()'s exact return shape — a caller could pass a
        ## hand-built partial dict (e.g. while gating only on F1_overall).
        ## Direct indexing would raise KeyError and lose the ENTIRE row
        ## (model, langs, epochs, checkpoint_path too) over one missing
        ## metric. Falling back to "" logs an incomplete-but-present row
        ## instead, which is recoverable; a missing row is not.
        "F1_ORG": scores.get("F1_ORG", ""),
        # Pull the ORGANIZATION F1 score, same empty-string-if-missing fallback.
        "F1_LOC": scores.get("F1_LOC", ""),
        # Pull the LOCATION F1 score, same fallback behavior.
        "F1_overall": scores.get("F1_overall", ""),
        # Pull the overall micro-F1 score, same fallback behavior.
        "checkpoint_path": checkpoint_path,
        # Store the checkpoint path argument (empty string if none given).
        "notes": notes,
        # Store the free-text notes argument (empty string if none given).
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure the results directory exists before writing; dirname(path) gets
    # just the folder part, and exist_ok=True means no error if it's already there.
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    # Decide whether a CSV header row needs to be written: true if the file
    # doesn't exist yet, OR it exists but is completely empty (0 bytes) —
    # both cases mean there's no header present yet.
    ## WHY check file SIZE and not just existence: os.makedirs just above
    ## guarantees the directory exists, but a file can exist on disk with 0
    ## bytes — e.g. from `touch`, from a previous run that crashed between
    ## creating the file and writing to it, or from the `.gitkeep`-style
    ## placeholder pattern used elsewhere in this repo. `not os.path.exists`
    ## alone would miss that case: it would see the file "exists" and skip
    ## writeheader(), so the very first real row would be appended into a
    ## headerless file — every downstream pandas.read_csv(path) call would
    ## then silently treat that first data row AS the header.
    ## WHY check it before opening in "a" mode rather than after: appending
    ## is a single pass — if the size were checked after a partial write
    ## began, the file would no longer be reliably empty/non-empty.
    with open(path, "a", newline="", encoding="utf-8") as f:
        # Open in append mode ("a") so existing rows are preserved and this
        # run is added after them; newline="" is required by the csv module
        # to avoid extra blank lines on Windows; encoding="utf-8" supports
        # Indic-script text that might appear in `notes`.
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        # Create a CSV writer that writes dict rows, using COLUMNS to fix
        # both which keys are expected and the exact column order in the file.
        if write_header:
            # Only write the header line the very first time the file is created.
            writer.writeheader()
            # Write the column names as the first row of the CSV.
        writer.writerow(row)
        # Write this run's data as the next row in the file.
