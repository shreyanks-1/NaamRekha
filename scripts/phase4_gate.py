"""Phase 4 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): the transfer matrix is filled (MuRIL trained on hi,
tested on hi/mr/bn) plus in-language runs for mr and bn.

This script checks that those rows exist for all three seeds, come from one
clean execution of the Phase 4 notebook, use the pinned Phase 3 checkpoints and
recipe, and are reported in the docs. It does not retrain or re-score any model.

Run from the repo root:  uv run python scripts/phase4_gate.py
"""
# Module docstring: Phase 4 studies cross-lingual transfer — how well a
# MuRIL model trained only on Hindi (hi) performs when evaluated on Marathi
# (mr) and Bengali (bn) without any in-language training data, compared to
# models trained directly on each target language. This gate verifies the
# full "transfer matrix" of required (train_lang, test_lang) combinations
# was actually run, for every seed, reusing the exact pinned Hindi
# checkpoints produced back in Phase 3.

from __future__ import annotations
# Enables postponed evaluation of type hints for modern generic annotations.

import csv
# Standard library csv module: reads results.csv and parses notebook-logged
# CSV-like lines.
import json
# Standard library json module: parses the notebook (.ipynb) JSON structure.
import os
# Standard library os module: path joining/resolution and existence checks.
import re
# Standard library re module: validates checkpoint URLs against a fixed pattern.
import sys
# Standard library sys module: extends the import path for `src`.
from collections import Counter
# Counter: used to compare the results.csv rows against the notebook's
# printed rows as multisets, regardless of order.
from statistics import mean, stdev
# mean: averages F1 scores across seeds for reporting in the doc-matching
# check and the transfer-gap summary.
# stdev: computes the sample standard deviation of the transfer gap across
# seeds, to show how consistent the gap is.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repo root is two directories above this script (scripts/ -> repo root).
sys.path.insert(0, REPO_ROOT)
# Make the repo root importable so `from src import ...` resolves correctly.

from src import COLUMNS, RESULTS_PATH  # noqa: E402
# Import the shared results-CSV schema and default path; `# noqa: E402`
# suppresses the expected lint warning from the necessary sys.path fix above.

MODEL = "MuRIL"
# Phase 4 only concerns the MuRIL model (the Phase 3 winner carried forward).
SOURCE = "hi"
# Hindi is the single source language all transfer experiments train from.
TARGETS = ["mr", "bn"]
# Marathi and Bengali are the two target languages transfer is tested on.
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase4-transfer.ipynb")
# Path to the Phase 4 notebook whose execution is being audited.
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase4", "README.md")
# Path to the Phase 4 write-up doc, expected to report mean F1 per transfer pair.
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
# The four F1 score columns checked for validity on every row.
REQUIRED = [(SOURCE, SOURCE), *[(SOURCE, t) for t in TARGETS], *[(t, t) for t in TARGETS]]
# Build the complete list of (train_lang, test_lang) pairs this gate
# requires to exist: hi->hi (the already-verified Phase 3 baseline,
# reproduced here as a sanity check), hi->mr and hi->bn (the zero-shot
# cross-lingual transfer cells), and mr->mr and bn->bn (the in-language
# baselines transfer performance is compared against).
HI_MODELS = {
    # A hardcoded registry of the exact Phase 3 Hindi checkpoints — one per
    # seed — that Phase 4 must reuse unchanged, each entry being
    # (huggingface repo id, pinned commit SHA, the Phase 3 F1 score that
    # checkpoint originally achieved on Hindi gold).
    "42": ("priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s42_bfd254ed",
           "de5204680ed4a8ab1ac842847df6866a6e2c9251", 0.8274137068534269),
    # Seed 42's checkpoint repo/commit and its originally-measured Hindi F1.
    "43": ("priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s43_223b5780",
           "f15d615b7d760d23af6c5aa111675f0d6be39c3d", 0.824173091270828),
    # Seed 43's checkpoint repo/commit and its originally-measured Hindi F1.
    "44": ("priyadarshi-su/naamrekha-muril_hi_n300000_e3_lr2e-05_s44_618532f6",
           "aa2f28136b49859a460217190b85ebedfc625e32", 0.8174346201743462),
    # Seed 44's checkpoint repo/commit and its originally-measured Hindi F1.
}
SEEDS = list(HI_MODELS)
# The three seeds this gate checks, derived directly from HI_MODELS' keys
# so the two stay in sync automatically.
## WHY derive SEEDS from HI_MODELS instead of writing SEEDS = ["42", "43",
## "44"] as its own separate literal: if these were two independent lists,
## editing one (e.g. adding a 4th seed to HI_MODELS when a new Phase 3
## checkpoint is pinned) without remembering to also update the other would
## leave the gate silently checking the WRONG set of seeds — either
## skipping the new seed entirely, or crashing with a KeyError looking up a
## seed in HI_MODELS that SEEDS mentions but HI_MODELS doesn't have.
## Deriving one from the other makes that class of edit-one-forget-the-other
## bug structurally impossible.
HI_REPRO_TOLERANCE = 0.001  # about two entities on the 1,929-entity Hindi gold set
# Allowed numerical drift when re-measuring the hi->hi F1 in Phase 4 versus
# the value originally recorded in Phase 3 — small floating-point/hardware
# nondeterminism is tolerated, but anything larger suggests a real
# discrepancy (e.g. a different checkpoint was accidentally loaded).
## WHY any nonzero tolerance at all, instead of requiring the two F1 numbers
## to match EXACTLY: reloading the same checkpoint weights on (possibly)
## different GPU hardware or a different library version doesn't guarantee
## bit-identical floating-point results — non-deterministic GPU kernels,
## different batch ordering, or minor numeric-precision differences can
## shift the score by a tiny amount even when the actual predictions are
## unchanged on all but a couple of entities. WHY exactly 0.001 and not
## looser: the comment ties it to a concrete, checkable scale ("about two
## entities" out of 1,929) — loose enough to absorb that harmless noise,
## tight enough that loading a genuinely WRONG checkpoint (which would
## typically shift dozens or hundreds of entity predictions, not one or
## two) still fails this check.
SUBSET_SIZE = "300000"
# The fixed training subset size (300,000 examples) every Phase 4 run must use.
EPOCHS = "3"
# The fixed epoch count every Phase 4 run must use.
CHECKPOINT_URL = re.compile(r"https://huggingface\.co/[^/\s]+/[^/\s]+/tree/[0-9a-f]{40}")
# Regex matching a Hugging Face URL pinned to an exact 40-character commit
# SHA, same pattern used in phase3_gate.py, ensuring checkpoints referenced
# here are also immutable.


def check(cond: bool, msg: str) -> None:
    # Shared assertion helper (same pattern as the other gate scripts).
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    # Print PASS/FAIL alongside the check's description.
    if not cond:
        # On failure...
        raise SystemExit(f"Gate failed: {msg}")
        # ...halt the script immediately with a clear message.


def parse_notes(notes: str) -> dict[str, str]:
    # Parses the "key=value; key=value; ..." notes string format into a dict.
    return dict(part.split("=", 1) for part in notes.split("; ") if "=" in part)
    # Split on "; " to get each key=value chunk, keep only chunks containing
    # "=", split each on the first "=" only (maxsplit=1) to avoid truncating
    # values containing "=", and build the dict from the resulting pairs.


def notebook_logged_rows(nb: dict) -> list[tuple[str, ...]]:
    """Rows the notebook printed after log_result (lines starting with MODEL,)."""
    # Docstring: extracts every CSV-row-like line the notebook printed that
    # corresponds to a logged MuRIL result, across all its code cells.
    lines = []
    # Accumulator for matching stdout lines found across all cells.
    for cell in nb["cells"]:
        # Iterate every cell in the notebook.
        text = "".join("".join(out.get("text", [])) for out in cell.get("outputs", [])
                       if out.get("output_type") == "stream" and out.get("name") == "stdout")
        # Gather this cell's stdout-stream output text, same pattern as
        # phase3_gate.py's notebook_stdout: join each matching output's
        # "text" chunk list, then join those joined outputs together.
        lines.extend(l for l in text.splitlines() if l.startswith(MODEL + ","))
        # Split this cell's combined stdout into lines, keep only the ones
        # starting with "MuRIL," (i.e. CSV rows for this model), and append
        # them to the running list.
    return [tuple(row) for row in csv.reader(lines)]
    # Parse every collected line as CSV (csv.reader handles the whole list
    # of lines at once here, unlike phase3_gate.py's single-line-at-a-time
    # approach), converting each parsed row list into a tuple for hashability.


def main() -> None:
    # Entry point running every Phase 4 gate check in sequence.
    print("1. Results sheet")
    # Section header: basic structural checks on results.csv and the Phase
    # 4 row count.
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    # Confirm results.csv exists before reading it.
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        # Open the results file for reading.
        reader = csv.DictReader(f)
        # Parse rows into dicts keyed by header column names.
        header = reader.fieldnames
        # Capture the detected header.
        rows = list(reader)
        # Materialize every data row.
    check(header == COLUMNS, "header matches the fixed columns")
    # Confirm the header matches the fixed project schema.
    phase4 = [r for r in rows if r["model"] == MODEL and "phase=4" in r["notes"]]
    # Filter to rows that are MuRIL runs explicitly tagged "phase=4" in
    # their notes — this tag is what distinguishes Phase 4's transfer runs
    # from Phase 3's own MuRIL rows (which share the same model name).
    check(not any("DEBUG" in r["notes"] for r in phase4), "no DEBUG rows committed")
    # None of the Phase 4 rows should be debug/throwaway runs.
    check(len(phase4) == len(REQUIRED) * len(SEEDS), f"{len(REQUIRED) * len(SEEDS)} Phase 4 rows logged ({len(phase4)} found)")
    # The total row count must be exactly (number of required transfer-matrix
    # cells) times (number of seeds) — e.g. 5 required pairs x 3 seeds = 15
    # rows — catching both missing and duplicate rows at a glance.
    ## WHY check the TOTAL count here, when section 2 below already checks
    ## `len(found) == 1` for every individual required cell: those per-cell
    ## checks can only ever catch a missing or duplicated row WITHIN the
    ## required set — they say nothing about whether some EXTRA, unexpected
    ## row also got tagged "phase=4" (e.g. a stray debug run that wasn't
    ## caught by the "DEBUG" substring convention, or a typo'd extra pair).
    ## If every required cell has exactly one row AND the total still came
    ## out higher than expected, this check is what flags that something
    ## extra and unaccounted-for is mixed into the Phase 4 results, which
    ## the per-cell loop alone would silently let through.

    print("\n2. Transfer matrix + in-language runs, per seed")
    # Section header: validates every individual (seed, train, test)
    # combination in detail.
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    # A lookup table from (seed, train_lang, test_lang) to that cell's full
    # results row, built up as we go, reused later for the doc-matching and
    # transfer-gap reporting sections.
    for seed in SEEDS:
        # Iterate each of the three required seeds.
        for train, test in REQUIRED:
            # Iterate each required (train_lang, test_lang) pair for this seed.
            name = f"seed {seed} {train}->{test}"
            # Build a short human-readable identifier for this specific cell,
            # used in all its check() messages below.
            found = [r for r in phase4 if r["train_lang"] == train and r["test_lang"] == test
                     and parse_notes(r["notes"]).get("seed") == seed]
            # Filter the Phase 4 rows down to ones matching this exact
            # (train_lang, test_lang, seed) combination.
            check(len(found) == 1, f"exactly one {name} row logged ({len(found)} found)")
            # Require exactly one matching row — zero means missing, more
            # than one means an ambiguous duplicate.
            r = found[0]
            # Unwrap the single matching row for convenience.
            for c in SCORE_COLS:
                # Validate each of the four F1 columns for this row.
                check(0.0 < float(r[c]) <= 1.0, f"{name} {c} = {float(r[c]):.4f} is a valid entity F1")
                # F1 must be strictly positive (a genuine run, not a
                # degenerate all-wrong baseline) and at most 1.0.
            check(r["subset_size"] == SUBSET_SIZE and r["epochs"] == EPOCHS, f"{name} uses n=300,000 and 3 epochs")
            # Confirm this run used the fixed training-set size and epoch count.
            check(bool(CHECKPOINT_URL.fullmatch(r["checkpoint_path"])), f"{name} checkpoint_path is a pinned Hugging Face link")
            # Confirm the checkpoint link is pinned to an exact commit SHA.
            ## WHY this branches into two entirely different kinds of check
            ## (exact-checkpoint-identity below vs. recipe-only further down)
            ## instead of one uniform check for every cell: hi->hi/mr/bn
            ## cells and mr->mr/bn->bn cells represent fundamentally
            ## different operations. The hi->* cells are ZERO-SHOT eval —
            ## no training happens in Phase 4 at all, they just reload an
            ## already-existing Phase 3 checkpoint and run inference on a
            ## new test language — so the only meaningful check is "is this
            ## the SAME exact weights as Phase 3." The in-language cells
            ## involve genuinely NEW training runs with no prior checkpoint
            ## to compare identity against, so the meaningful check instead
            ## becomes "was the training recipe/hyperparameters the same
            ## fixed setup used everywhere else" — a different question
            ## because it's a different kind of cell.
            if train == SOURCE:
                # For any cell trained on Hindi (the hi->hi, hi->mr, and
                # hi->bn cells), the checkpoint must be the EXACT Phase 3
                # checkpoint for that seed — these cells don't retrain,
                # they just evaluate the existing Hindi model on a new test set.
                repo, revision, _ = HI_MODELS[seed]
                # Look up this seed's registered Phase 3 (repo, commit) pair,
                # ignoring the third tuple element (original F1) here.
                check(r["checkpoint_path"] == f"https://huggingface.co/{repo}/tree/{revision}",
                      f"{name} uses that seed's pinned Phase 3 checkpoint")
                # Confirm the row's checkpoint_path exactly reconstructs the
                # registered Phase 3 URL for this seed — i.e. no retraining
                # happened and no wrong/different checkpoint was substituted.
            else:
                # For the in-language cells (mr->mr, bn->bn), a genuinely
                # new model was trained on that target language, so instead
                # of checking a specific checkpoint identity, verify the
                # training setup itself followed the fixed recipe.
                notes = parse_notes(r["notes"])
                # Parse this row's notes for the hyperparameter fields.
                check(notes.get("gpus") == "1" and notes.get("effective_bs") == "32" and int(notes.get("warmup_steps", 0)) > 0,
                      f"{name} used one GPU, effective batch 32, and warmup")
                # Confirm single-GPU, effective batch size 32, and a
                # non-zero warmup schedule — the same fixed setup used
                # throughout the project, so in-language runs are
                # comparable to the transfer runs.
            cells[(seed, train, test)] = r
            # Store this validated row in the lookup table for later reuse.
        expected = HI_MODELS[seed][2]
        # For this seed, retrieve the originally-recorded Phase 3 Hindi F1
        # score (the third element of the HI_MODELS tuple).
        measured = float(cells[(seed, SOURCE, SOURCE)]["F1_overall"])
        # Retrieve the hi->hi F1 score as measured/re-logged in this Phase 4
        # notebook run, for the same seed.
        check(abs(measured - expected) <= HI_REPRO_TOLERANCE,
              f"seed {seed} hi->hi {measured:.4f} is within {HI_REPRO_TOLERANCE} of Phase 3 score {expected:.4f}")
        # Confirm the two numbers (Phase 3's original score vs. Phase 4's
        # re-measurement of the same checkpoint) agree within the small
        # allowed tolerance — a strong signal the exact same model weights
        # were actually reloaded and evaluated identically, not a different
        # or corrupted checkpoint.
    commits = {parse_notes(r["notes"]).get("git") for r in phase4}
    # Collect the distinct git commit hashes recorded across every Phase 4 row.
    check(len(commits) == 1 and None not in commits, f"all Phase 4 rows come from one code commit ({sorted(map(str, commits))})")
    # Require exactly one distinct commit hash, and that it's never
    # missing/None (which would mean some row's notes lacked a "git=" field
    # entirely) — ensures every Phase 4 result was produced by one
    # consistent version of the code.
    ## WHY the explicit `None not in commits` half of this check matters
    ## (not just `len(commits) == 1` on its own): parse_notes(...).get("git")
    ## returns None whenever a row's notes string has no "git=" field at
    ## all. If EVERY Phase 4 row happened to be missing that field (e.g. a
    ## future refactor of the logging code dropped it by accident),
    ## `commits` would be the single-element set `{None}` — and
    ## `len(commits) == 1` would be TRUE, passing this check despite the
    ## fact that no commit was actually recorded for any row at all. The
    ## explicit None check closes that loophole: "everyone agrees" and
    ## "everyone agrees on NOTHING" must not look the same to this gate.

    print("\n3. One clean Phase 4 notebook execution")
    # Section header: verifies the Phase 4 notebook ran top-to-bottom once
    # from a fresh kernel, and that its printed output matches results.csv exactly.
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    # Confirm the notebook file exists.
    with open(NOTEBOOK, encoding="utf-8") as f:
        # Open the notebook (a JSON file) for reading.
        nb = json.load(f)
        # Parse it into a Python dict.
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    # Collect each code cell's Jupyter execution-count number, in cell order.
    check(counts == list(range(1, len(counts) + 1)), "every code cell ran once, in order, from a fresh kernel")
    # Confirm the execution counts form an unbroken 1, 2, 3, ... sequence —
    # proof of a single clean top-to-bottom run.
    logged = [tuple(r[c] for c in COLUMNS) for r in phase4]
    # Build tuples of each Phase 4 results.csv row's values, in COLUMNS order.
    check(Counter(notebook_logged_rows(nb)) == Counter(logged), "results rows exactly match the notebook's printed rows")
    # Compare the notebook's printed rows against the results.csv rows as
    # multisets (via Counter), confirming every logged row was genuinely
    # printed by this exact notebook execution, with matching counts.

    print("\n4. Phase 4 doc")
    # Section header: confirms the Phase 4 write-up doc reports the correct
    # mean F1 for every required transfer-matrix cell.
    check(os.path.exists(PHASE_DOC), f"{os.path.relpath(PHASE_DOC, REPO_ROOT)} exists")
    # Confirm the Phase 4 doc file exists.
    with open(PHASE_DOC, encoding="utf-8") as f:
        # Open the doc as UTF-8 text.
        doc = f.read()
        # Read its full contents into a string.
    for train, test in REQUIRED:
        # Check each required transfer-matrix cell's reported mean F1.
        f1 = f"{mean(float(cells[(s, train, test)]['F1_overall']) for s in SEEDS):.4f}"
        # Compute the mean overall F1 across all three seeds for this
        # (train, test) pair, formatted to 4 decimal places as prose would quote it.
        check(f1 in doc, f"doc reports {train}->{test} mean F1 {f1}")
        # Confirm that exact formatted mean score string appears somewhere
        # in the doc's text.

    print("\n  transfer gap (in-language - zero-shot, F1_overall), paired by seed:")
    # Section header for a purely informational summary (not a pass/fail
    # check) showing how much better in-language training does versus
    # zero-shot transfer from Hindi, for each target language.
    ## WHY this section only prints, with no check()/pass-fail gate on the
    ## gap's sign or size (e.g. no "assert gap > 0"): this gate's job is to
    ## verify the DATA is complete, traceable, and trustworthy — not to
    ## enforce a particular research conclusion. A negative or small gap
    ## (zero-shot transfer matching or beating in-language training) would
    ## be a genuinely interesting, publishable finding, not a bug to reject.
    ## Turning this into a hard pass/fail check would conflate "the
    ## experiment was run correctly" with "the experiment confirmed what we
    ## expected," and would make the gate fail on a legitimate result.
    for t in TARGETS:
        # Compute and print the transfer gap for each target language (mr, bn).
        gaps = [float(cells[(s, t, t)]["F1_overall"]) - float(cells[(s, SOURCE, t)]["F1_overall"]) for s in SEEDS]
        # For each seed, compute (in-language F1 on target t) minus
        # (zero-shot F1 transferring from Hindi to t) — a positive gap means
        # in-language training outperforms zero-shot transfer, as expected.
        per_seed = ", ".join(f"{s}: {g:+.4f}" for s, g in zip(SEEDS, gaps))
        # Build a readable per-seed breakdown string, e.g. "42: +0.0531, 43:
        # +0.0498, 44: +0.0512"; the `+` format flag always shows the sign.
        print(f"    {SOURCE} -> {t}: mean {mean(gaps):+.4f}, sd {stdev(gaps):.4f} ({per_seed})")
        # Print the mean gap across seeds, its standard deviation (showing
        # how consistent the gap is across random seeds), and the per-seed breakdown.
    print("\nPhase 4 gate: PASSED")
    # Reaching here means every check() above succeeded.


if __name__ == "__main__":
    # Only run main() when this file is executed directly as a script.
    main()
    # Run the gate.
