"""Phase 1 gate check. Uses only the shared code in src/.

Gate (docs/01-roadmap.md): load Hindi, print the IOB2 tag distribution, and
score a dummy all-O prediction with seqeval. Also checks that Marathi and
Bengali load with the same format, since Phase 4 depends on them.

Run from the repo root:  uv run python scripts/phase1_gate.py
"""
# Module docstring: documents this script's purpose as an automated
# "gate" (pass/fail sanity check) that validates Phase 1 infrastructure
# (data loading, label map, evaluator, results logger) before later phases
# build on top of it.

from __future__ import annotations
# Enables postponed evaluation of type hints (PEP 563) for modern annotation
# syntax used in this file's helper function signatures.

import csv
# Standard library csv module: used later to read back the results CSV file
# written by log_result, to verify its contents match expectations.
import os
# Standard library os module: used for path manipulation (computing the
# repo root) and filesystem checks.
import sys
# Standard library sys module: used to modify the Python import path so the
# `src` package (one directory up) can be imported when running this script directly.
import tempfile
# Standard library tempfile module: used to create a throwaway temp
# directory for testing log_result without touching the real results.csv.

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# __file__ is this script's path; abspath() makes it absolute, the first
# dirname() strips the filename to get `scripts/`, and the second dirname()
# goes up one more level to the repo root. Inserting that at the front of
# sys.path (position 0) makes `import src` resolve correctly regardless of
# the current working directory this script is launched from.

from src import (  # noqa: E402
    # The `# noqa: E402` comment suppresses a linter warning about imports
    # not being at the very top of the file — unavoidable here since the
    # sys.path fix above must run first for this import to succeed.
    COLUMNS,
    # The fixed list of results-CSV column names, used below to verify the
    # logger writes the expected header.
    LABELS,
    # The canonical ordered list of BIO tag strings, used to check the
    # label contract hasn't drifted.
    SPLITS,
    # The tuple of valid dataset split names ("train", "val", "test").
    evaluate,
    # The F1-scoring function, used to sanity-check the evaluator itself.
    id2label,
    # int -> tag string lookup, used to verify round-trip consistency with label2id.
    label2id,
    # tag string -> int lookup, used to verify round-trip consistency with id2label.
    load_split,
    # The dataset split loader, exercised against Hindi/Marathi/Bengali.
    log_result,
    # The CSV results-appending function, exercised against a temp file.
    tag_distribution,
    # The tag-frequency counter, used to print/verify label distributions.
)

EXPECTED_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
# A hardcoded copy of the expected label order, independent of src/data.py's
# own LABELS constant, so this gate can detect if someone accidentally
# changes the label contract in data.py without updating this check.
HI_TEST_ROWS = 864  # 867 rows in the file minus 3 empty ones (docs/phase1)
# The expected exact sentence count for the Hindi test split after empty
# sentences are dropped by load_split — a regression check against a known,
# previously-verified number.


def check(cond: bool, msg: str) -> None:
    # Small shared assertion helper: prints a PASS/FAIL line for every gate
    # condition and aborts the whole script immediately on first failure.
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    # Print a human-readable status line; the ternary picks "PASS" or "FAIL"
    # text based on whether the condition held, followed by the description.
    if not cond:
        # If the condition is false, this gate check has failed.
        raise SystemExit(f"Gate failed: {msg}")
        # Immediately terminate the script with a non-zero exit status and
        # an explanatory message, since later checks likely depend on this one.


def check_rows(lang: str, split: str, rows) -> None:
    # Reusable validation routine applied to every (lang, split) pair's
    # loaded rows, checking the basic data-integrity invariants expected of
    # any split.
    check(len(rows) > 0, f"{lang}/{split}: {len(rows):,} sentences loaded")
    # Verify at least one sentence was loaded; the message reports the exact
    # count with thousands separators for readability.
    check(all(w for w, _ in rows), f"{lang}/{split}: no empty sentences")
    # For every (words, tags) row, `w` must be a non-empty list (truthy);
    # `all(...)` over a generator confirms this holds for every single row,
    # i.e. load_split's empty-sentence filtering actually worked.
    check(all(len(w) == len(t) for w, t in rows), f"{lang}/{split}: words and tags align")
    # Confirm each row's token list and tag list have matching lengths —
    # critical, since a token/tag misalignment would silently corrupt training.
    unknown = set(tag_distribution(rows)) - set(LABELS)
    # Compute the tag frequency counter for this split, take just its keys
    # (the distinct tags seen), and subtract the set of known LABELS — any
    # remainder is an unexpected/unknown tag that shouldn't exist.
    check(not unknown, f"{lang}/{split}: only known tags (extra: {sorted(unknown)})")
    # Pass only if `unknown` is empty; otherwise report exactly which
    # unexpected tags were found, sorted for stable/readable output.


def print_distribution(rows) -> None:
    # Prints a human-readable table of tag counts and percentages for a
    # given set of (words, tags) rows.
    counts = tag_distribution(rows)
    # Compute the Counter of tag -> frequency across all rows.
    total = sum(counts.values())
    # Sum all tag occurrences to use as the denominator for percentages.
    for label in LABELS:
        # Iterate labels in the fixed canonical order (not dict/Counter
        # iteration order) so the printed table is consistently ordered.
        n = counts.get(label, 0)
        # Look up this label's count, defaulting to 0 if it never occurred
        # in this split (e.g. a rare entity type missing from a small split).
        print(f"    {label:<6} {n:>8,}  ({n / total:6.2%})")
        # Print the label left-aligned in a 6-char field, the count
        # right-aligned in an 8-char field with thousands separators, and
        # the percentage of the total formatted as a 6-wide percentage.


def main() -> None:
    # Entry point that runs every Phase 1 gate check in sequence, printing
    # numbered section headers as it goes.
    print("1. Label map")
    # Section header for the first group of checks: validating the label
    # vocabulary itself.
    check(LABELS == EXPECTED_LABELS, "label order matches the Phase 1 contract")
    # Confirm the live LABELS constant from src/data.py still exactly equals
    # the hardcoded EXPECTED_LABELS defined above (same tags, same order).
    check(all(id2label[label2id[x]] == x for x in LABELS), "label2id / id2label round-trip")
    # For every label string x, converting it to an id via label2id and then
    # back to a string via id2label should return the original x — this
    # verifies the two dicts are true inverses of each other.

    print("\n2. Hindi test split")
    # Section header: checks specific to the Hindi ("hi") test split.
    hi_test = load_split("hi", "test")
    # Load the Hindi test split from the Naamapadam dataset via the shared loader.
    check_rows("hi", "test", hi_test)
    # Run the generic row-integrity checks (non-empty, aligned, known tags only).
    check(len(hi_test) == HI_TEST_ROWS, f"hi/test has {HI_TEST_ROWS} sentences")
    # Confirm the exact sentence count matches the previously-recorded
    # expected value, catching any unexpected change in the upstream dataset
    # or in the loader's filtering logic.
    print("  Tag distribution (hi/test):")
    # Sub-header before printing the distribution table.
    print_distribution(hi_test)
    # Print the per-tag counts/percentages for this split.

    print("\n3. Evaluator")
    # Section header: checks that the seqeval-based evaluator behaves
    # correctly on two known edge cases (all-wrong and all-correct).
    gold = [tags for _, tags in hi_test]
    # Extract just the tag sequences (discarding the words) from the Hindi
    # test split to use as the "ground truth" for evaluator sanity checks.
    all_o = [["O"] * len(tags) for tags in gold]
    # Build a dummy "predicts everything as Outside" baseline: for each gold
    # sentence, produce a same-length list of all "O" tags — a trivial
    # baseline predictor that should score zero on every entity metric.
    scores = evaluate(gold, all_o)
    # Score the all-O dummy predictions against gold using the shared evaluator.
    print(f"  all-O scores: {scores}")
    # Print the resulting score dict for visibility/debugging.
    check(all(v == 0.0 for v in scores.values()), "all-O prediction scores 0.0 everywhere")
    # Since the dummy predictor never predicts any entity, every F1 score
    # (overall and per type) should be exactly 0.0 — verifies evaluate()
    # correctly penalizes a predictor that finds no entities.
    perfect = evaluate(gold, gold)
    # Score the gold labels against themselves — a "perfect prediction" case.
    print(f"  gold-vs-gold scores: {perfect}")
    # Print those scores for visibility/debugging.
    check(all(v == 1.0 for v in perfect.values()), "gold vs itself scores 1.0 everywhere")
    # A predictor that exactly reproduces gold labels should score a perfect
    # 1.0 on every metric — verifies evaluate() isn't miscounting or
    # penalizing correct predictions.

    print("\n4. Results logger (temp file, real results.csv untouched)")
    # Section header: checks log_result's CSV-writing behavior in isolation,
    # explicitly avoiding any writes to the project's real results.csv.
    with tempfile.TemporaryDirectory() as tmp:
        # Create a temporary directory that is automatically deleted when
        # this `with` block exits, regardless of success or failure.
        path = os.path.join(tmp, "results.csv")
        # Build a throwaway results.csv path inside the temp directory.
        for _ in range(2):
            # Call log_result twice in a row to verify the header is written
            # only once (on the first call) and not duplicated on the second.
            log_result(model="dummy-all-O", train_lang="none", test_lang="hi",
                       subset_size=0, epochs=0, scores=scores, path=path)
            # Log a result row using the all-O dummy scores computed above,
            # explicitly pointing at the temp `path` instead of the real
            # project results.csv (via the path= keyword override).
        with open(path, encoding="utf-8", newline="") as f:
            # Re-open the temp CSV file for reading, to inspect what was
            # actually written; newline="" matches the writer's convention
            # to avoid platform-specific line-ending surprises.
            rows = list(csv.reader(f))
            # Parse the file into a list of rows (each row a list of string cells).
        check(rows[0] == COLUMNS, "header matches the fixed columns")
        # The very first row should be exactly the COLUMNS header list.
        check(len(rows) == 3, "header written once, one row per call")
        # Total rows should be 3: one header row + two data rows (one per
        # log_result call) — confirms the header-writing logic correctly
        # fires only when the file is new/empty, not on every call.

    print("\n5. All splits for hi / mr / bn")
    # Section header: broader smoke test confirming every split loads
    # correctly for all three languages this project cares about.
    for lang in ("hi", "mr", "bn"):
        # Iterate Hindi, Marathi, and Bengali — the three languages Phase 4's
        # transfer-learning experiments will need.
        for split in SPLITS:
            # Iterate every valid split name ("train", "val", "test").
            if lang == "hi" and split == "test":
                continue  # already checked above
                # Skip re-checking Hindi's test split since section 2 above
                # already loaded and validated it in detail.
            check_rows(lang, split, load_split(lang, split))
            # Load this (lang, split) combination fresh and run the generic
            # row-integrity checks on it.
        if lang != "hi":
            # For the non-Hindi languages, additionally print their test-set
            # tag distribution (Hindi's was already printed in section 2).
            print(f"  Tag distribution ({lang}/test):")
            # Sub-header naming which language's distribution follows.
            print_distribution(load_split(lang, "test"))
            # Load that language's test split again and print its tag distribution.

    print("\nPhase 1 gate: PASSED")
    # If execution reaches this line, every `check()` call above succeeded
    # (since any failure would have raised SystemExit and stopped the script).


if __name__ == "__main__":
    # Standard Python idiom: only run main() when this file is executed
    # directly as a script (e.g. `python scripts/phase1_gate.py`), not when
    # it's imported as a module elsewhere.
    main()
    # Invoke the gate's entry point.
