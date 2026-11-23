"""Phase 3 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): both MuRIL and IndicBERT are scored on Hindi gold;
MuRIL clearly beats the BiLSTM-CRF baseline (0.8038, logged in Phase 2); the
scaling curve (F1 vs subset_size) is logged in the results sheet and a
working subset size is chosen from it.

This script checks that every Phase 3 row comes from one top-to-bottom
notebook execution of one commit, uses the fixed training setup, links to a
pinned checkpoint, and agrees with the chosen size in the docs. It does not
retrain or re-score any model.

Run from the repo root:  uv run python scripts/phase3_gate.py
"""
# Module docstring: this is the most involved gate in the project — rather
# than just checking the final numbers, it forensically verifies that the
# logged Phase 3 results all came from one single, reproducible, top-to-bottom
# notebook run (same git commit, same hyperparameters, pinned checkpoints),
# and that the chosen subset size was picked by a well-defined rule rather
# than cherry-picked.

from __future__ import annotations
# Enables postponed evaluation of type hints for the modern generic
# annotation syntax used throughout this file.

import csv
# Standard library csv module: used to read results.csv and to re-parse
# CSV-like lines recovered from the notebook's saved stdout.
import json
# Standard library json module: used to parse the notebook (.ipynb) file.
import os
# Standard library os module: used for path joins/resolution and existence checks.
import re
# Standard library re module: used for regex matching against checkpoint
# URLs, notebook-printed lines, and doc/markdown text.
import sys
# Standard library sys module: used to extend the import path for `src`.
from collections import Counter, defaultdict
# Counter: used to compare multisets of CSV rows (results.csv rows vs.
# notebook-printed rows) irrespective of order, catching missing/extra/duplicated rows.
# defaultdict: used to group scaling-curve F1 scores by subset size without
# manually initializing each size's list.
from statistics import mean
# mean: computes the arithmetic average, used to average F1 across multiple
# random seeds at each subset size on the scaling curve.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Compute the repository root two directories above this script's own path.
sys.path.insert(0, REPO_ROOT)
# Make the repo root importable so `from src import ...` below works
# regardless of the current working directory.

from src import COLUMNS, RESULTS_PATH  # noqa: E402
# Import the shared results-CSV schema (COLUMNS) and default file path
# (RESULTS_PATH); `# noqa: E402` suppresses the expected "import not at top"
# lint warning caused by the necessary sys.path fix above.

LANG = "hi"
# This gate only concerns Hindi ("hi") gold-data experiments.
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase3-muril-indicbert.ipynb")
# Path to the Phase 3 notebook whose saved execution is being audited.
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase3", "README.md")
# Path to the Phase 3 write-up doc, expected to report the same numbers as results.csv.
ROADMAP = os.path.join(REPO_ROOT, "docs", "01-roadmap.md")
# Path to the overall project roadmap doc, which also records the Phase 3 outcome.
DECISIONS = os.path.join(REPO_ROOT, "docs", "06-decisions.md")
# Path to the project's running decisions log, expected to record the chosen subset size.
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
# The four F1 score columns validated for being sane values.
BASELINE_MODEL = "BiLSTM-CRF"
# The Phase 2 model name used as the baseline MuRIL must "clearly beat".
PHASE_MODELS = ("MuRIL", "IndicBERT")
# The two transformer models evaluated in Phase 3.
REQUIRED_NOTES = ("seed", "lr", "effective_bs", "gpus", "warmup_steps", "config", "val_F1", "git", "torch", "transformers")
# The set of key=value fields that every Phase 3 results row's "notes"
# column must record, for full reproducibility (random seed, learning rate,
# effective batch size, GPU count, warmup schedule, config name, validation
# F1, the exact git commit, and the torch/transformers library versions used).
MIN_SCALING_POINTS = 3  # at least this many distinct subset sizes logged for MuRIL
# Minimum number of distinct training-set sizes required to call it a
# genuine "scaling curve" rather than just one or two data points.
MIN_MARGIN = 0.01  # "clearly beats" requires at least this much headroom over baseline
# The minimum F1 improvement over the Phase 2 baseline required to count as
# "clearly" beating it, rather than a statistically-noisy tie.
CURVE_EPOCHS = "3"
# The fixed epoch count used for every scaling-curve run, so sizes are
# compared on equal footing (a string since results.csv stores CSV text).
CURVE_LR = "2e-05"
# The fixed learning rate used for every scaling-curve run, same reasoning
# (controlling this variable so only subset_size differs across the curve).
HANDOFF_SEED = "42"
# The specific random seed designated as the "handoff" run — the one whose
# checkpoint actually gets carried forward into later phases.
REQUIRED_SEEDS = {"42", "43", "44"}  # the chosen size must be confirmed on all of these
# The chosen subset size must have been run with all three of these seeds,
# to confirm the result isn't a lucky roll of one particular seed.
SIZE_TOLERANCE = 0.01  # same rule as the notebook's Config cell
# How much worse (in F1) a smaller subset size is allowed to be compared to
# the best-performing size, while still being "close enough" to prefer for
# efficiency — must match the notebook's own selection logic exactly.
COLLAPSE_F1 = 0.1
# A validation F1 below this threshold is considered a "collapsed"
# (failed/diverged) training run, which disqualifies that subset size
# regardless of its mean score.
CHECKPOINT_URL = re.compile(r"https://huggingface\.co/[^/\s]+/[^/\s]+/tree/[0-9a-f]{40}")
# Regex matching a Hugging Face Hub URL pinned to an exact 40-character
# commit SHA (e.g. ".../tree/<40 hex chars>") — ensures checkpoints are
# referenced immutably rather than by a mutable branch name like "main".
RETIRED_SCORES = ("0.8330", "0.8343", "0.0807")  # pre-restart runs, must not be reported
# Specific F1 values from an earlier, since-discarded set of runs (e.g.
# before some bug was fixed and training was restarted) that must never
# reappear in the current docs, as a regression check against accidentally
# re-reporting stale numbers.


def check(cond: bool, msg: str) -> None:
    # Same shared assertion helper pattern as the other gate scripts.
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    # Print PASS/FAIL plus the description of what was checked.
    if not cond:
        # On failure...
        raise SystemExit(f"Gate failed: {msg}")
        # ...stop the script immediately with a clear message.


def parse_notes(notes: str) -> dict[str, str]:
    # Parses the semicolon-separated "key=value; key=value; ..." format used
    # in the results.csv "notes" field back into a dictionary.
    return dict(part.split("=", 1) for part in notes.split("; ") if "=" in part)
    # Split the notes string on "; " to get each "key=value" chunk; for each
    # chunk that actually contains an "=" (guarding against stray/malformed
    # parts), split on the FIRST "=" only (maxsplit=1, via split("=", 1)) so
    # values that themselves contain "=" aren't truncated; feed all the
    # resulting (key, value) tuples into dict(...) to build the final mapping.


def read_text(path: str) -> str:
    # Small helper to read an entire text file's contents in one call,
    # reducing repetition across the many file reads in this script.
    with open(path, encoding="utf-8") as f:
        # Open as UTF-8 text (docs may contain non-ASCII characters).
        return f.read()
        # Return the whole file content as a single string.


def notebook_stdout(nb: dict) -> list[str]:
    """Printed lines from every code cell, in cell order."""
    # Docstring: reconstructs, in original execution order, every line of
    # text the notebook printed to standard output across all its code cells.
    lines = []
    # Accumulator list for all printed lines, across the whole notebook.
    for cell in nb["cells"]:
        # Iterate every cell (code and markdown) in notebook order.
        text = "".join("".join(out.get("text", [])) for out in cell.get("outputs", [])
                       if out.get("output_type") == "stream" and out.get("name") == "stdout")
        # For this cell, gather only the outputs that are "stream" type
        # outputs from "stdout" (as opposed to stderr, display_data/images,
        # or error tracebacks); each matching output's "text" field is
        # itself a list of string chunks, so the inner "".join(...) merges
        # those chunks, and the outer "".join(...) merges across multiple
        # matching outputs within the same cell into one string.
        lines.extend(text.splitlines())
        # Split this cell's combined stdout text into individual lines and
        # append them to the running list, preserving overall cell order.
    return lines
    # Return every stdout line from the whole notebook, in the order it was printed.


def choose_subset_size(curve_rows: list[dict[str, str]]) -> int:
    """Smallest size with no collapsed seed whose mean val F1 is within tolerance of the best."""
    # Docstring: documents the exact selection rule being re-implemented
    # here — it must exactly match whatever rule the notebook itself used,
    # so this function can independently verify the notebook's stated choice.
    val_by_size = defaultdict(list)
    # Maps each subset size (int) to a list of validation-F1 scores (one per
    # seed) observed at that size; defaultdict(list) auto-creates empty
    # lists for new keys so no manual initialization is needed.
    for r in curve_rows:
        # Iterate every scaling-curve row (already filtered to fixed
        # epochs/lr by the caller).
        val_by_size[int(r["subset_size"])].append(float(r["parsed_notes"]["val_F1"]))
        # Convert this row's subset_size to int (to use as a clean dict key)
        # and append its validation F1 (parsed out of the notes field by
        # parse_notes, stashed under "parsed_notes" by the caller) to that
        # size's list of scores.
    means = {n: mean(f1s) for n, f1s in val_by_size.items()}
    # For each subset size, compute the mean validation F1 across all its
    # recorded seeds — smooths out single-seed noise.
    best = max(means.values())
    # The highest mean validation F1 achieved at any subset size — the
    # ceiling that smaller sizes are compared against.
    for n in sorted(val_by_size):
        # Iterate subset sizes in ascending order, so the first one that
        # qualifies is genuinely the *smallest* qualifying size.
        if min(val_by_size[n]) >= COLLAPSE_F1 and means[n] >= best - SIZE_TOLERANCE:
            # Qualify this size only if BOTH: (a) its worst-performing seed
            # didn't collapse (stayed at or above COLLAPSE_F1, ruling out a
            # diverged/failed run skewing the mean upward by luck elsewhere),
            # and (b) its mean F1 is within SIZE_TOLERANCE of the best mean
            # observed at any size (i.e. "good enough" compared to the best).
            return n
            # Return immediately — since sizes are iterated ascending, this
            # is the smallest size satisfying the rule, favoring efficiency.
    raise SystemExit("Gate failed: no subset size satisfies the selection rule")
    # If no size ever satisfies both conditions, the whole gate fails
    # outright, since Phase 3 cannot proceed without a valid chosen size.


def main() -> None:
    # Entry point running every Phase 3 gate check section by section.
    print("1. Results sheet")
    # Section header: basic checks on the results.csv file's structure and
    # presence of the required baseline/phase rows.
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    # Confirm the results file exists before reading it.
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        # Open results.csv for reading.
        reader = csv.DictReader(f)
        # Parse rows into dicts keyed by the header's column names.
        header = reader.fieldnames
        # Capture the detected header for schema validation.
        rows = list(reader)
        # Materialize every data row.
    check(header == COLUMNS, "header matches the fixed columns")
    # Confirm the CSV's header matches the project's fixed column schema.

    baseline_runs = [r for r in rows if r["model"] == BASELINE_MODEL and r["train_lang"] == LANG and r["test_lang"] == LANG
                     and "DEBUG" not in r["notes"]]
    # Filter to genuine (non-debug) BiLSTM-CRF Hindi-to-Hindi runs — these
    # represent the Phase 2 baseline this gate needs to compare against.
    check(len(baseline_runs) >= 1, f"{BASELINE_MODEL} {LANG}->{LANG} baseline row present (Phase 2)")
    # Require at least one such baseline row to exist.
    baseline_f1 = max(float(r["F1_overall"]) for r in baseline_runs)
    # Take the best (highest) overall F1 among the baseline runs, in case
    # multiple baseline runs were logged.
    print(f"  baseline (BiLSTM-CRF) Hindi gold F1 = {baseline_f1:.4f}")
    # Print the baseline figure for visibility.

    phase_rows = [r for r in rows if r["model"] in PHASE_MODELS and r["train_lang"] == LANG and r["test_lang"] == LANG
                  and "phase=4" not in r["notes"]]
    # Filter to MuRIL/IndicBERT Hindi-to-Hindi rows, explicitly excluding
    # any rows tagged "phase=4" in their notes (since Phase 4's transfer
    # experiments reuse the same model names but belong to a different gate).
    check(len(phase_rows) >= 1, "Phase 3 rows logged")
    # Require at least one Phase 3 row to exist.
    check(not any("DEBUG" in r["notes"] for r in phase_rows), "no DEBUG rows committed")
    # None of the Phase 3 rows should be marked DEBUG — only final, genuine
    # runs should be committed to the shared results sheet.

    print("\n2. Run setup and checkpoints")
    # Section header: per-row validation of training configuration and checkpoint pinning.
    for r in phase_rows:
        # Validate every individual Phase 3 row.
        r["parsed_notes"] = notes = parse_notes(r["notes"])
        # Parse this row's notes string into a dict and also stash it back
        # onto the row dict itself (under a new "parsed_notes" key) so later
        # sections (like choose_subset_size) can reuse it without re-parsing.
        name = f"{r['model']} n={r['subset_size']} e={r['epochs']} seed={notes.get('seed')} lr={notes.get('lr')}"
        # Build a short human-readable identifier for this specific run, used
        # in all of this row's check() messages below.
        missing = [k for k in REQUIRED_NOTES if k not in notes]
        # Find which of the required reproducibility fields are absent from
        # this row's parsed notes.
        check(not missing, f"{name}: notes record the run setup (missing: {missing or 'none'})")
        # Pass only if nothing is missing; otherwise report exactly which
        # fields are absent.
        check(notes["gpus"] == "1" and notes["effective_bs"] == "32", f"{name}: single GPU, effective batch 32")
        # Enforce the fixed training setup: exactly one GPU and an effective
        # batch size of 32, so all runs are comparable.
        check(int(notes["warmup_steps"]) > 0, f"{name}: warmup applied")
        # A learning-rate warmup schedule must have been used (non-zero warmup steps).
        for c in SCORE_COLS:
            # Validate each of the four F1 score columns for this row.
            check(0.0 <= float(r[c]) <= 1.0, f"{name}: {c} is a valid entity F1")
            # F1 must fall within the mathematically valid [0, 1] range.
        check(bool(CHECKPOINT_URL.fullmatch(r["checkpoint_path"])), f"{name}: checkpoint_path is a pinned Hugging Face link")
        # The checkpoint_path must exactly match the CHECKPOINT_URL regex —
        # i.e. a Hugging Face URL pinned to a specific 40-char commit SHA,
        # not a mutable branch reference.
    commits = {r["parsed_notes"]["git"] for r in phase_rows}
    # Collect the distinct git commit hashes recorded across all Phase 3 rows.
    check(len(commits) == 1, f"all Phase 3 rows come from one code commit ({sorted(commits)})")
    # Require every single Phase 3 row to have been produced from exactly
    # the same git commit — guards against mixing results from different,
    # possibly inconsistent versions of the training code.

    print("\n3. One clean notebook execution")
    # Section header: verifies the notebook was run top-to-bottom, once,
    # from a fresh kernel (not cells re-run out of order or re-executed
    # multiple times, which could produce misleading/duplicated output).
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    # Confirm the notebook file exists.
    nb = json.loads(read_text(NOTEBOOK))
    # Parse the notebook's JSON content into a Python dict.
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    # Collect each code cell's "execution_count" (Jupyter's per-cell run
    # counter), in cell order, skipping markdown cells which have none.
    check(counts == list(range(1, len(counts) + 1)), "every code cell ran once, in order, from a fresh kernel")
    # In a notebook executed cleanly top-to-bottom exactly once from a fresh
    # kernel, the execution counts must be exactly 1, 2, 3, ... in cell
    # order with no gaps, repeats, or out-of-order numbers — any deviation
    # means cells were re-run, skipped, or reordered.
    stdout = notebook_stdout(nb)
    # Extract every printed stdout line from the notebook, in execution order.
    printed = [tuple(row) for row in csv.reader(l for l in stdout if l.startswith(tuple(m + "," for m in PHASE_MODELS)))]
    # Filter stdout lines down to ones that look like CSV rows for either
    # "MuRIL," or "IndicBERT," (built dynamically from PHASE_MODELS), then
    # feed that filtered line generator through csv.reader to parse each
    # line into a list of fields, converting each parsed row into a tuple
    # (tuples are hashable, needed for the Counter comparison below).
    logged = [tuple(r[c] for c in COLUMNS) for r in phase_rows]
    # Build the equivalent tuples from the results.csv Phase 3 rows, pulling
    # each column value in the same COLUMNS order, so both sides are
    # directly comparable tuple-for-tuple.
    check(Counter(printed) == Counter(logged),
          f"results rows exactly match the notebook's printed rows ({len(logged)} logged, {len(printed)} printed)")
    # Compare the two lists as multisets (via Counter) rather than ordered
    # lists — this verifies every row in results.csv was actually printed by
    # the notebook (and vice versa, with correct counts), regardless of
    # ordering differences, catching any row added, removed, or duplicated
    # by hand.

    print("\n4. MuRIL scaling curve and size choice")
    # Section header: verifies the scaling-curve experiment and the
    # notebook's stated subset-size decision.
    muril = [r for r in phase_rows if r["model"] == "MuRIL"]
    # Filter Phase 3 rows down to just the MuRIL runs (IndicBERT is handled separately below).
    curve = [r for r in muril if r["epochs"] == CURVE_EPOCHS and r["parsed_notes"]["lr"] == CURVE_LR]
    # Further filter to runs using the exact fixed epoch count and learning
    # rate designated for the scaling curve, so only directly-comparable
    # runs (varying only in subset_size and seed) are included.
    sizes = sorted({int(r["subset_size"]) for r in curve})
    # Collect the distinct subset sizes present on the curve, sorted ascending.
    check(len(sizes) >= MIN_SCALING_POINTS, f"at least {MIN_SCALING_POINTS} distinct subset sizes on the curve ({sizes})")
    # Require enough distinct sizes to constitute a real curve, not just one
    # or two points.
    chosen = choose_subset_size(curve)
    # Independently (re-)derive which subset size the selection rule picks,
    # using this script's own implementation of that rule.
    stated = [int(m.group(1)) for l in stdout if (m := re.fullmatch(r"working subset size: (\d+)", l))]
    # Scan the notebook's stdout for a line exactly matching "working subset
    # size: <digits>" (the notebook's own announcement of its chosen size);
    # the walrus operator `:=` assigns the match object to `m` inline so it
    # can be reused in the `if` condition and the list comprehension body
    # without matching twice; extract the captured number as an int for
    # every such line found (expected to be exactly one).
    check(stated == [chosen], f"notebook's chosen size {stated} matches the validation-F1 rule ({chosen})")
    # Confirm the notebook stated exactly one chosen size, and that it
    # matches what this script's independent re-derivation of the rule produces.
    chosen_rows = [r for r in curve if int(r["subset_size"]) == chosen]
    # Filter the curve rows down to just those at the chosen subset size.
    seeds = {r["parsed_notes"]["seed"] for r in chosen_rows}
    # Collect the distinct random seeds that were run at the chosen size.
    check(REQUIRED_SEEDS <= seeds, f"chosen size ran on seeds {sorted(REQUIRED_SEEDS)} (found {sorted(seeds)})")
    # Using set subset comparison (`<=`), require that every seed in
    # REQUIRED_SEEDS ({"42","43","44"}) was actually run at the chosen size
    # — confirms the chosen size's result is seed-robust, not a one-off.
    handoff = [r for r in chosen_rows if r["parsed_notes"]["seed"] == HANDOFF_SEED]
    # Identify the specific run, at the chosen size, using the designated
    # "handoff" seed (42) — this is the checkpoint later phases will reuse.
    check(len(handoff) == 1, f"exactly one handoff row at n={chosen}, seed {HANDOFF_SEED}")
    # There must be exactly one such row — ambiguity (zero or multiple)
    # would make "the" handoff checkpoint undefined.
    handoff = handoff[0]
    # Unwrap the single-element list into the row dict itself for convenience below.
    handoff_f1 = float(handoff["F1_overall"])
    # Extract the handoff run's overall F1 score as a float.
    margin = handoff_f1 - baseline_f1
    # Compute how much better (or worse) the handoff MuRIL run is compared
    # to the Phase 2 BiLSTM-CRF baseline.
    check(margin >= MIN_MARGIN, f"handoff MuRIL F1 {handoff_f1:.4f} beats baseline {baseline_f1:.4f} by >= {MIN_MARGIN} (margin {margin:.4f})")
    # Enforce the roadmap's "clearly beats" requirement: the margin must
    # meet or exceed the minimum threshold defined by MIN_MARGIN.
    verified = [m for l in stdout if (m := re.match(r"handoff verified: (\S+)@([0-9a-f]{40}) ", l))]
    # Scan stdout for a line confirming the handoff checkpoint was actually
    # reloaded from storage (not just trusted blindly) and re-scored,
    # capturing the Hugging Face repo id and the exact commit SHA it was
    # reloaded from.
    check(len(verified) == 1 and handoff["checkpoint_path"] == f"https://huggingface.co/{verified[0][1]}/tree/{verified[0][2]}",
          "handoff checkpoint was reloaded from its pinned revision and re-scored")
    # Require exactly one such verification line, and that the repo id + SHA
    # it reports reconstruct exactly the same URL stored in the row's
    # checkpoint_path — proving the logged checkpoint link is the one that
    # was actually reloaded and verified to reproduce the score, not just
    # asserted.

    print("\n5. IndicBERT comparison")
    # Section header: verifies IndicBERT was also evaluated, specifically at
    # the chosen subset size, for a fair head-to-head comparison with MuRIL.
    indicbert = [r for r in phase_rows if r["model"] == "IndicBERT" and int(r["subset_size"]) == chosen]
    # Filter to IndicBERT rows at exactly the chosen subset size.
    check(len(indicbert) == 1, f"one IndicBERT row at the chosen size (n={chosen})")
    # Require exactly one such row to exist.

    print("\n6. Phase 3 docs")
    # Section header: cross-checks that every relevant doc file quotes
    # consistent numbers, and that no stale/retired scores leak back in.
    check(os.path.exists(PHASE_DOC), f"{os.path.relpath(PHASE_DOC, REPO_ROOT)} exists")
    # Confirm the Phase 3 doc exists.
    doc = read_text(PHASE_DOC)
    # Read the Phase 3 doc's full text.
    roadmap = read_text(ROADMAP)
    # Read the full roadmap doc's text.
    roadmap_phase = roadmap[roadmap.index("## Phase 3"):roadmap.index("## Phase 4")]
    # Slice out just the "## Phase 3" section of the roadmap, from its
    # heading up to (but not including) the next "## Phase 4" heading, so
    # checks below only look at text relevant to this phase.
    check(f"{handoff_f1:.4f}" in doc, f"README reports the handoff F1 {handoff_f1:.4f}")
    # Confirm the Phase 3 README quotes the handoff F1 score (formatted to 4
    # decimal places, matching typical prose reporting style).
    doc_size = re.search(r"chosen working subset size:\**\s*([\d,]+)", doc, re.IGNORECASE)
    # Search the doc for a line like "Chosen working subset size: **12,345**"
    # (allowing optional markdown bold asterisks and comma-separated
    # thousands), case-insensitively, capturing the numeric part.
    check(bool(doc_size) and int(doc_size.group(1).replace(",", "")) == chosen, f"README states the chosen size {chosen:,}")
    # Require that such a line was found AND that its number (with commas
    # stripped before int-parsing) exactly equals the independently-derived
    # chosen size.
    check(f"{handoff_f1:.4f}" in roadmap_phase and f"n={chosen:,}" in roadmap_phase, "roadmap reports the same F1 and size")
    # Confirm the roadmap's Phase 3 section also quotes both the same F1
    # score and the chosen size (formatted as "n=12,345").
    decision = re.search(r"phase 3 working subset size[^\n]*?(\d[\d,]{3,})", read_text(DECISIONS), re.IGNORECASE)
    # Search the project's decisions log for a line mentioning "phase 3
    # working subset size" followed somewhere later on the same line by a
    # multi-digit number, case-insensitively.
    check(bool(decision) and int(decision.group(1).replace(",", "")) == chosen, "decisions log records the chosen size")
    # Require that such a decision entry exists and records the correct chosen size.
    stale = [s for s in RETIRED_SCORES if s in doc or s in roadmap_phase]
    # Check whether any of the specifically-retired, pre-restart F1 values
    # still appear anywhere in the Phase 3 doc or roadmap section.
    check(not stale, f"docs do not report retired pre-restart scores ({stale or 'none'})")
    # Fail if any retired score leaked back into the current documentation.

    print(f"\n  MuRIL Hindi gold F1 (handoff, n={chosen}): {handoff_f1:.4f}")
    # Print a final human-readable summary of the key headline result.
    print(f"  Baseline (BiLSTM-CRF): {baseline_f1:.4f}")
    # Print the baseline figure alongside it for easy comparison.
    print("\nPhase 3 gate: PASSED")
    # Reaching here means every check() above succeeded.


if __name__ == "__main__":
    # Only run main() when this file is executed directly as a script.
    main()
    # Run the gate.
