"""Phase 5 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): LLM entity-F1 (zero-shot and few-shot) is in the
results sheet with the model name and sample size recorded, and one page of
concrete error examples is written.

Beyond that, this script checks that the two rows come from one clean
top-to-bottom notebook run of one commit on all 864 gold sentences, that the
model revision is pinned, that malformed output was counted (not dropped),
and that the F1 in the sheet can be reproduced by re-scoring the saved
per-sentence predictions with the shared evaluator. It does not run any model.

Run from the repo root:  uv run python scripts/phase5_gate.py
"""
# Module docstring: Phase 5 evaluates a general-purpose LLM (prompted, not
# fine-tuned) on the NER task, under both zero-shot and few-shot prompting.
# This is the most thorough gate in the project: beyond checking the logged
# F1 numbers, it independently RE-DERIVES the F1 score from scratch by
# replaying every saved raw model output through the committed parser
# (phase5_parser.py) and the shared evaluator — so a wrong or stale number
# in results.csv would be caught even if the notebook's own printed output
# somehow agreed with it.

from __future__ import annotations
# Enables postponed evaluation of type hints for modern generic annotations.

import csv
# Standard library csv module: reads results.csv and parses CSV-like lines
# recovered from notebook stdout.
import hashlib
# Standard library hashlib module: used to compute a SHA-1 hash of the
# committed parser file, to verify the exact parser code used to produce
# the logged results hasn't silently drifted since.
import json
# Standard library json module: parses the notebook (.ipynb) file and the
# line-delimited JSON predictions file.
import os
# Standard library os module: path joins/resolution and existence checks.
import re
# Standard library re module: validates checkpoint URLs and scans notebook
# stdout / doc text for specific patterns.
import sys
# Standard library sys module: extends the import path for `src` and the
# sibling `phase5_parser`/`test_phase5_parser` modules.
from collections import Counter
# Counter: compares results.csv rows against notebook-printed rows as
# multisets, same pattern used in the other gate scripts.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repo root is two directories above this script.
sys.path.insert(0, REPO_ROOT)
# Make the repo root importable so `from src import ...` resolves.
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
# ALSO add the scripts/ directory itself to the import path, since this
# gate needs to import sibling modules that live there directly
# (phase5_parser and, inside main(), test_phase5_parser) as plain top-level
# modules rather than as a package.

from phase5_parser import parse_llm_output  # noqa: E402
# Import the actual parsing function under test/audit — used later to
# independently re-parse every saved raw model output and confirm it
# reproduces the saved predictions exactly.
from src import COLUMNS, RESULTS_PATH, evaluate, load_split  # noqa: E402
# Import the shared results-CSV schema/path, the F1 evaluator, and the
# dataset split loader — used to re-score predictions against the real gold
# test split as an independent check. `# noqa: E402` on both import lines
# suppresses the expected "import not at top" warning from the necessary
# sys.path fixes above.

LANG = "hi"
# Phase 5 evaluates Hindi only.
FULL_TEST = 864
# The exact size of the Hindi gold test split (matching Phase 1's recorded
# HI_TEST_ROWS) — Phase 5 requires scoring on the FULL set, not a sample.
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase5-llm-prompting.ipynb")
# Path to the Phase 5 LLM-prompting notebook being audited.
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase5", "README.md")
# Path to the Phase 5 write-up doc.
ERROR_DOC = os.path.join(REPO_ROOT, "docs", "phase5", "error_analysis.md")
# Path to the dedicated one-page error-analysis doc this phase must produce.
ROADMAP = os.path.join(REPO_ROOT, "docs", "01-roadmap.md")
# Path to the overall project roadmap doc.
PREDICTIONS = os.path.join(REPO_ROOT, "results", "phase5", "final_predictions.jsonl")
# Path to the saved line-delimited JSON file holding every individual
# sentence's raw model output, parsed prediction, and gold tags — the
# ground truth this gate uses to independently re-derive the F1 score.
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
# The four F1 score columns validated for being sane and reproducible.
MODES = ("zero-shot", "few-shot")
# The two prompting strategies Phase 5 must cover: no examples in the
# prompt (zero-shot) versus a handful of example sentence/answer pairs
# (few-shot).
REQUIRED_NOTES = ("phase", "mode", "k_shot", "test_n", "seed", "model", "revision", "prompt", "parser", "config",
                  "malformed", "malformed_rate", "git", "torch", "transformers")
# The key=value fields every Phase 5 results row's notes must record:
# which phase/mode this run is, how many few-shot examples (k_shot), how
# many sentences were tested (test_n), the random seed, which LLM and
# exact model revision was used, which prompt/parser/config version,
# malformed-output count and rate, the git commit, and library versions.
CHECKPOINT_URL = re.compile(r"https://huggingface\.co/([^/\s]+/[^/\s]+)/tree/([0-9a-f]{40})")
# Regex matching a Hugging Face URL pinned to an exact 40-character commit
# SHA, with two capture groups: the "owner/repo" part and the commit hash
# itself, so both can be individually cross-checked against the notes fields.
ERROR_KINDS = ("missed", "boundary", "spurious", "type confusion", "format")
# The five categories of NER error the error-analysis doc is required to
# discuss: entities the model missed entirely, span-boundary mistakes
# (e.g. including/excluding a word), spans invented that aren't real
# entities, entities tagged with the wrong type, and output-format problems.
RETIRED_SCORES = ("0.5335", "0.5741")   # the earlier 200-sentence gold-test draft, not a result of this phase
# F1 values from an earlier, smaller-scale draft evaluation (only 200
# sentences, since superseded by the full 864-sentence run) that must never
# reappear in the current docs — same "no stale numbers" pattern as the
# other gate scripts' RETIRED_SCORES.


def check(cond: bool, msg: str) -> None:
    # Shared assertion helper (same pattern as every other gate script).
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    # Print PASS/FAIL alongside the check's description.
    if not cond:
        # On failure...
        raise SystemExit(f"Gate failed: {msg}")
        # ...halt immediately with a clear message.


def parse_notes(notes: str) -> dict[str, str]:
    # Parses the "key=value; key=value; ..." notes string into a dict, same
    # implementation as the other phase gate scripts.
    return dict(part.split("=", 1) for part in notes.split("; ") if "=" in part)
    # Split on "; ", keep only parts containing "=", split each on the first
    # "=" only, build the dict from the resulting pairs.


def read_text(path: str) -> str:
    # Small helper to read an entire file's text contents in one call.
    with open(path, encoding="utf-8") as f:
        # Open as UTF-8 text.
        return f.read()
        # Return the whole content as one string.


def notebook_stdout(nb: dict) -> list[str]:
    # Extracts every line the notebook printed to stdout, across all cells,
    # in original execution order — same implementation pattern as the
    # equivalent helper in phase3_gate.py.
    lines = []
    # Accumulator for every stdout line found.
    for cell in nb["cells"]:
        # Iterate every cell in the notebook.
        text = "".join("".join(out.get("text", [])) for out in cell.get("outputs", [])
                       if out.get("output_type") == "stream" and out.get("name") == "stdout")
        # Gather and join this cell's stdout-stream output text (ignoring
        # stderr, error tracebacks, and non-text outputs like images).
        lines.extend(text.splitlines())
        # Split into individual lines and append to the running list.
    return lines
    # Return every stdout line from the whole notebook, in order.


def main() -> None:
    # Entry point running every Phase 5 gate check in sequence.
    print("1. Parser tests (no GPU)")
    # Section header: before trusting ANY of the parser's output, first
    # confirm the parser itself passes its full unit test suite.
    import test_phase5_parser as tp  # noqa: E402
    # Import the sibling test module (found via the scripts/ sys.path entry
    # added above) as `tp`; the import happens here inside main() rather
    # than at module top-level since it's only needed for this one check.
    # `# noqa: E402` suppresses the "import not at top of file" lint warning.

    tests = [f for n, f in sorted(vars(tp).items()) if n.startswith("test_") and callable(f)]
    # Scan the test module's namespace (vars(tp) returns its globals as a
    # dict) for every name starting with "test_" that's callable, sorted
    # alphabetically for a deterministic run order — same discovery pattern
    # test_phase5_parser.py itself uses internally for `_run_all`.
    for t in tests:
        # Run every discovered test function.
        t()
        # Call it directly (not wrapped in try/except here, unlike
        # test_phase5_parser.py's own runner) — if ANY parser test fails,
        # it raises an AssertionError that propagates up and crashes this
        # whole gate script immediately, which is the desired strict behavior.
    check(len(tests) >= 20, f"all {len(tests)} parser tests pass")
    # Sanity-check that a substantial number of tests actually ran (guards
    # against an import/discovery bug silently finding zero tests and
    # falsely appearing to "pass" by doing nothing).

    print("\n2. Results sheet")
    # Section header: structural checks on results.csv and locating the two
    # required LLM rows.
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    # Confirm results.csv exists.
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
    llm = [r for r in rows if r["model"].startswith("LLM-") and r["test_lang"] == LANG]
    # Filter to rows whose model name starts with "LLM-" (this phase's
    # naming convention, e.g. "LLM-zero-shot (...)") and that were tested on Hindi.
    check(not any("DEBUG" in r["notes"] for r in llm), "no DEBUG rows committed")
    # None of these should be debug/throwaway runs.
    by_mode = {}
    # Will map each prompting mode ("zero-shot"/"few-shot") to its single
    # results row, for convenient lookup in later sections.
    for mode in MODES:
        # Locate exactly one row for each required mode.
        found = [r for r in llm if r["model"].startswith(f"LLM-{mode} (")]
        # Filter to rows whose model name starts with "LLM-<mode> (" — the
        # parenthesis presumably introduces the specific model name that
        # follows in the full model string.
        check(len(found) == 1, f"exactly one {mode} row logged ({len(found)} found)")
        # Require exactly one such row per mode.
        by_mode[mode] = found[0]
        # Store it for later use.
    check(len(llm) == 2, "no other LLM rows in the sheet (earlier drafts removed)")
    # The total LLM row count must be EXACTLY 2 (one per mode) — if it's
    # more, some earlier draft/experimental row wasn't cleaned up before committing.

    print("\n3. Rows follow docs/03-conventions.md and the Phase 5 rules")
    # Section header: detailed per-row validation of the run configuration
    # recorded in each mode's notes field.
    for mode, r in by_mode.items():
        # Validate each of the two mode rows (zero-shot, few-shot).
        r["parsed"] = n = parse_notes(r["notes"])
        # Parse this row's notes into a dict, also stashing it back onto the
        # row (under "parsed") for reuse in later sections; `n` is a
        # convenience alias for the same dict.
        missing = [k for k in REQUIRED_NOTES if k not in n]
        # Find which required reproducibility fields are absent.
        check(not missing, f"{mode}: notes record the setup (missing: {missing or 'none'})")
        # Require nothing to be missing.
        check(n["phase"] == "5" and n["mode"] == mode, f"{mode}: phase=5 and mode matches the model name")
        # Confirm the notes explicitly self-identify as phase 5, and that
        # the recorded "mode" value matches the mode this row was filed
        # under (catches e.g. a copy-paste mistake where a few-shot row's
        # notes still say mode=zero-shot).
        for c in SCORE_COLS:
            # Validate each F1 score column.
            check(0.0 < float(r[c]) <= 1.0, f"{mode}: {c} = {float(r[c]):.4f} is a valid entity F1")
            # Must be a strictly positive, at-most-1.0 F1 value.
        check(r["train_lang"] == "none" and r["subset_size"] == "0" and r["epochs"] == "0",
              f"{mode}: no training (train_lang=none, subset_size=0, epochs=0)")
        # Phase 5 uses a pretrained/prompted LLM with NO fine-tuning at
        # all — confirm the row correctly records zero training (unlike
        # Phases 2-4, which all trained a model).
        used, total = (int(x) for x in n["test_n"].split("/"))
        # The notes' "test_n" field is formatted as "used/total" (e.g.
        # "864/864"); split on "/" and convert both halves to int.
        check(used == total == FULL_TEST, f"{mode}: scored on all {FULL_TEST} gold sentences (test_n={n['test_n']})")
        # Both the "used" and "total" counts must equal the FULL 864-sentence
        # test set — i.e. every single gold sentence was scored, with none
        # skipped and no sampling/subsetting.
        m = CHECKPOINT_URL.fullmatch(r["checkpoint_path"])
        # Attempt to match the checkpoint_path against the pinned-Hugging-
        # Face-URL pattern, capturing the repo id and commit SHA if it matches.
        check(bool(m), f"{mode}: checkpoint_path is a pinned Hugging Face URL")
        # Require the checkpoint_path to actually match that pattern.
        check(m.group(1) == n["model"] and m.group(2) == n["revision"], f"{mode}: URL matches model and revision in notes")
        # Cross-check that the URL's captured repo id and commit SHA
        # exactly match the separately-recorded "model" and "revision"
        # fields in the notes — these are two independent places the same
        # information is recorded, and they must agree.
        check(n["malformed"].isdigit() and abs(int(n["malformed"]) / FULL_TEST - float(n["malformed_rate"])) < 1e-4,
              f"{mode}: malformed output counted ({n['malformed']} of {FULL_TEST}, rate {n['malformed_rate']})")
        # Confirm the "malformed" count is a plain non-negative integer
        # string, AND that dividing it by FULL_TEST reproduces the
        # separately-recorded "malformed_rate" field to within a tiny
        # floating-point tolerance — again, two independently-recorded
        # numbers that must be mutually consistent.
    z, f_ = by_mode["zero-shot"]["parsed"], by_mode["few-shot"]["parsed"]
    # Pull out the parsed-notes dicts for both modes; `f_` (with a trailing
    # underscore) is used instead of `f` to avoid shadowing the common `f`
    # variable name used for open file handles elsewhere in this style of code.
    check(z["k_shot"] == "0" and int(f_["k_shot"]) >= 1, f"few-shot uses {f_['k_shot']} examples, zero-shot uses 0")
    # Zero-shot must record exactly zero few-shot examples; few-shot must
    # record at least one — confirms the two modes are genuinely configured differently.
    for k in ("model", "revision", "git", "prompt", "parser", "config", "seed"):
        # For every OTHER aspect of the setup that should be held constant
        # between the two modes (so the only real difference is the
        # presence/absence of few-shot examples)...
        check(z[k] == f_[k], f"both rows share {k} ({z[k][:12]})")
        # ...confirm the zero-shot and few-shot rows recorded the identical
        # value for that field; the message previews just the first 12
        # characters of the shared value for a readable status line.

    print("\n4. One clean notebook execution")
    # Section header: verifies the Phase 5 notebook ran cleanly top-to-
    # bottom once, without errors, and that its printed output matches
    # results.csv exactly.
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    # Confirm the notebook file exists.
    nb = json.loads(read_text(NOTEBOOK))
    # Parse the notebook's JSON structure.
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    # Collect each code cell's execution-count number, in cell order.
    check(counts == list(range(1, len(counts) + 1)), "every code cell ran once, in order, from a fresh kernel")
    # Confirm an unbroken 1, 2, 3, ... sequence (one clean top-to-bottom run).
    errors = [o for c in nb["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
    # Scan every cell's outputs for any output specifically marked as type
    # "error" (a Python traceback that occurred during execution).
    check(not errors, "no cell raised an error")
    # Require there to be none — any cell erroring during the "clean" run
    # would call the whole run's validity into question.
    stdout = notebook_stdout(nb)
    # Extract every stdout line printed by the notebook, in order.
    check(any(l.startswith("stage: final | debug: False") for l in stdout), "the executed run was STAGE=final, DEBUG=False")
    # Confirm at least one stdout line explicitly announces this was the
    # "final" (not a draft/debug) run with debug mode off — presumably the
    # notebook prints this banner near the top to self-document its own run mode.
    tests_line = [m for l in stdout if (m := re.search(r"(\d+)/(\d+) parser tests passed", l))]
    # Search stdout for the test-summary line test_phase5_parser.py's
    # `_run_all()` prints (e.g. "31/31 parser tests passed"), capturing both
    # numbers; the walrus operator assigns the match to `m` for reuse in the
    # comprehension's filter condition and its body.
    check(len(tests_line) == 1 and tests_line[0][1] == tests_line[0][2], "parser tests passed at the start of the run")
    # Confirm exactly one such summary line appears (the notebook ran the
    # parser's own test suite once, presumably near its start as a
    # self-check), and that its two captured numbers are equal (i.e. ALL
    # tests passed — "passed/total" with passed == total).
    commit = [m.group(1) for l in stdout if (m := re.search(r"\| commit: (\w+)", l))]
    # Search stdout for a line recording which git commit the notebook was
    # run from (e.g. "... | commit: abc1234"), capturing the commit identifier.
    check(commit == [z["git"]], f"notebook ran commit {commit}, rows record {z['git']}")
    # Confirm exactly one such commit was printed, and it matches the git
    # commit recorded in the results.csv rows' notes.
    printed = [tuple(r) for r in csv.reader(l for l in stdout if l.startswith("LLM-"))]
    # Filter stdout lines to ones starting with "LLM-" (this phase's CSV-row
    # printout prefix) and parse them as CSV, converting each parsed row to
    # a tuple.
    logged = [tuple(r[c] for c in COLUMNS) for r in by_mode.values()]
    # Build the equivalent tuples from the two results.csv rows, in COLUMNS order.
    check(Counter(printed) == Counter(logged), "results rows exactly match the notebook's printed rows")
    # Compare as multisets — every logged row must have genuinely been
    # printed by this exact notebook execution.
    check(any("reproduces the Phase 3 handoff score" in l for l in stdout),
          "MuRIL comparison reproduced Phase 3's 0.8274 on the same sentences")
    # Confirm the notebook also re-ran the Phase 3 MuRIL checkpoint on these
    # same 864 sentences (for a fair side-by-side comparison against the
    # LLM) and explicitly printed confirmation that it reproduced the
    # original Phase 3 handoff score — guarding against comparing the LLM
    # against a stale or different MuRIL number.

    print("\n5. Saved predictions reproduce the logged F1")
    # Section header: the deepest level of verification in this gate —
    # independently re-deriving the F1 score from the raw saved model
    # outputs, rather than trusting any number the notebook itself printed.
    check(os.path.exists(PREDICTIONS), f"{os.path.relpath(PREDICTIONS, REPO_ROOT)} exists")
    # Confirm the saved per-sentence predictions file exists.
    recs: dict[str, list[dict]] = {}
    # Will map each "system" name (e.g. "LLM zero-shot [...]" or "MuRIL
    # [...]") to the list of per-sentence prediction records for that system.
    with open(PREDICTIONS, encoding="utf-8") as f:
        # Open the predictions file (line-delimited JSON, i.e. one JSON
        # object per line) for reading.
        for line in f:
            # Process the file one line (one sentence's record) at a time.
            rec = json.loads(line)
            # Parse this line as a JSON object.
            recs.setdefault(rec["system"], []).append(rec)
            # Group records by their "system" field; setdefault creates a
            # fresh empty list the first time a new system name is seen.
    # The committed parser must be the one that produced the logged rows (hash taken on LF line endings,
    # because the Kaggle run read the file from a Linux clone).
    with open(os.path.join(REPO_ROOT, "scripts", "phase5_parser.py"), "rb") as f:
        # Open the actual committed parser source file in BINARY mode (not
        # text mode), since we need exact byte control over line endings for hashing.
        parser_sha = hashlib.sha1(f.read().replace(b"\r\n", b"\n")).hexdigest()[:8]
        # Read the whole file as raw bytes, normalize any Windows-style
        # CRLF line endings to Unix-style LF (b"\r\n" -> b"\n") BEFORE
        # hashing — necessary because the comment above explains the actual
        # training run happened on Kaggle, which checked out the repo on
        # Linux (LF line endings), so hashing this local checkout
        # byte-for-byte would give a different hash purely due to line-
        # ending differences introduced by git/Windows checkout settings,
        # even if the code content is identical; compute a SHA-1 hash of
        # the normalized bytes and take just its first 8 hex characters as
        # a short fingerprint.
    check(parser_sha == z["parser"], f"committed parser matches the logged hash ({parser_sha})")
    # Confirm the hash of the CURRENTLY committed parser file exactly
    # matches the hash recorded in the results row's notes at the time the
    # run was logged — proving the parser code hasn't been modified since
    # (if it had, re-running the checks below with today's parser wouldn't
    # actually prove anything about the originally-logged run).
    test_rows = load_split(LANG, "test")           # the real gold test split, from the shared loader
    # Load the actual Hindi gold test split fresh from the shared data
    # loader, to compare against what was saved in the predictions file
    # (guards against the saved file having stale or tampered words/gold tags).
    for mode, r in by_mode.items():
        # Verify both modes' saved predictions in turn.
        system = next((s for s in recs if s.startswith(f"LLM {mode} [")), None)
        # Find the system name in the predictions file that corresponds to
        # this mode, matching the "LLM <mode> [...]" naming prefix; `next(...,
        # None)` returns None if no matching system name is found at all.
        check(system is not None, f"{mode}: predictions saved")
        # Require that such a system was actually found.
        rs = sorted(recs[system], key=lambda x: x["idx"])
        # Sort this system's records by their "idx" field (presumably the
        # sentence's index in the test split), to guarantee a consistent,
        # position-matched order for the zip-based comparisons below.
        check(len(rs) == FULL_TEST, f"{mode}: {len(rs)} sentences saved")
        # Confirm exactly 864 sentence records were saved for this mode.
        check(all(len(x["pred"]) == len(x["gold"]) == len(x["words"]) for x in rs), f"{mode}: one tag per word everywhere")
        # For every saved record, its predicted-tags list, gold-tags list,
        # and words list must all be the exact same length — basic
        # internal-consistency check on the saved data itself.
        check(len(rs) == len(test_rows) and all(x["words"] == w and x["gold"] == tg for x, (w, tg) in zip(rs, test_rows)),
              f"{mode}: saved words and gold match the real test split")
        # Confirm the saved records' word tokens and gold tags EXACTLY
        # match the freshly-loaded real gold test split, sentence for
        # sentence in order — proving the saved predictions file wasn't
        # built against a different, stale, or corrupted version of the
        # dataset.
        check(all(parse_llm_output(x["raw"], x["words"]).tags == x["pred"] for x in rs),
              f"{mode}: raw outputs re-parse to the saved predictions")
        # THE key reproducibility check: for every saved record, re-run the
        # CURRENT parser on the saved RAW model output text, and confirm it
        # produces the EXACT SAME predicted tags that were saved — proving
        # the saved "pred" field genuinely came from parsing "raw" with
        # this exact parser (combined with the hash check above confirming
        # this is the same parser code used originally).
        scores = evaluate([x["gold"] for x in rs], [x["pred"] for x in rs])
        # Independently re-compute the F1 scores from scratch, directly
        # from the saved gold/pred tag lists, using the shared evaluator —
        # not trusting any score the notebook itself claims to have computed.
        for c in SCORE_COLS:
            # Compare each independently-recomputed score column...
            check(abs(scores[c] - float(r[c])) < 1e-9, f"{mode}: re-scored {c} = {scores[c]:.4f} matches the sheet")
            # ...against the value actually logged in results.csv, to
            # within a tiny floating-point tolerance — this is the
            # strongest possible proof that the logged F1 is correct and reproducible.
        n_bad = sum(bool(x["malformed"]) for x in rs)
        # Count how many saved records have a truthy "malformed" field.
        check(n_bad == int(r["parsed"]["malformed"]), f"{mode}: {n_bad} malformed outputs in the saved raw outputs match the sheet")
        # Confirm this independently-counted malformed-output count matches
        # the count recorded in the results.csv notes.
        check(all(x["pred"] == ["O"] * len(x["pred"]) for x in rs if x["malformed"]),
              f"{mode}: every malformed sentence was scored as all O (not dropped)")
        # For every record marked malformed, confirm its saved prediction
        # is indeed all-"O" tags (R1's documented behavior in
        # phase5_parser.py) — proving malformed sentences were scored as
        # wrong-everywhere rather than silently excluded from the F1
        # calculation (which would artificially inflate the score).
    check(any(s.startswith("MuRIL") for s in recs), "MuRIL predictions saved for the error comparison")
    # Confirm the predictions file also includes at least one MuRIL system
    # (the fine-tuned baseline from Phase 3/4, re-evaluated on these same
    # 864 sentences), needed for the side-by-side error analysis doc.

    print("\n6. Docs")
    # Section header: cross-checks that the written documentation reports
    # the correct numbers and covers the required error-analysis content.
    for path in (PHASE_DOC, ERROR_DOC):
        # Check both required doc files exist.
        check(os.path.exists(path), f"{os.path.relpath(path, REPO_ROOT)} exists")
        # Confirm existence of this specific doc path.
    doc, err_doc = read_text(PHASE_DOC), read_text(ERROR_DOC)
    # Read both docs' full text contents.
    for mode, r in by_mode.items():
        # Check the README reports each mode's results correctly.
        check(f"{float(r['F1_overall']):.4f}" in doc, f"README reports the {mode} F1 {float(r['F1_overall']):.4f}")
        # Confirm the README quotes this mode's overall F1 score (4 decimal
        # places, prose-style formatting).
        check(f"{r['parsed']['malformed']} malformed" in doc.lower() or f"malformed: {r['parsed']['malformed']}" in doc.lower(),
              f"README reports the {mode} malformed count ({r['parsed']['malformed']})")
        # Confirm the README also reports the malformed-output count for
        # this mode, accepting either of two common phrasings ("N
        # malformed" or "malformed: N"), matched case-insensitively.
    check(z["model"] in doc and z["revision"] in doc, "README names the model and its pinned revision")
    # Confirm the README names both the exact model used and its pinned
    # commit revision, for full transparency/reproducibility in the human-readable docs.
    check(str(FULL_TEST) in doc, "README states the test size")
    # Confirm the README states the test set size (864) explicitly.
    check(all(k in err_doc.lower() for k in ERROR_KINDS), f"error analysis covers: {', '.join(ERROR_KINDS)}")
    # Confirm the error-analysis doc mentions every required error category
    # (missed, boundary, spurious, type confusion, format), matched
    # case-insensitively as a simple substring check.
    check(len(re.findall(r"[ऀ-ॿ]+", err_doc)) >= 30, "error analysis has concrete Hindi examples")
    # `[ऀ-ॿ]` is the Unicode codepoint range for the Devanagari script block;
    # count how many separate runs of Devanagari characters appear in the
    # doc, requiring at least 30 — a proxy check ensuring the error
    # analysis contains genuine concrete Hindi example sentences/words, not
    # just abstract English discussion of error types.
    check(len(err_doc.splitlines()) <= 120, "error analysis stays about one page")
    # Enforce the roadmap's "one page of concrete error examples"
    # requirement by capping the doc at 120 lines — a rough proxy for "about
    # one page" of markdown text.
    stale = [s for s in RETIRED_SCORES if s in doc or s in err_doc]
    # Check whether either retired, pre-full-run F1 score still appears in
    # either doc.
    check(not stale, f"docs do not report the retired 200-sentence draft scores ({stale or 'none'})")
    # Fail if any retired score leaked into the current documentation.
    roadmap = read_text(ROADMAP)
    # Read the full roadmap doc's text.
    section = roadmap[roadmap.index("## Phase 5"):roadmap.index("## Phase 6")]
    # Slice out just the "## Phase 5" section of the roadmap, from its
    # heading up to (but not including) the next "## Phase 6" heading.
    check(all(f"{float(r['F1_overall']):.4f}" in section for r in by_mode.values()), "roadmap Phase 5 reports both F1 scores")
    # Confirm the roadmap's Phase 5 section quotes BOTH modes' overall F1 scores.

    print("\n  LLM Hindi gold F1 (" + z["model"] + ", all 864 sentences):")
    # Print a final human-readable summary header naming the specific LLM
    # evaluated and confirming the full test set size.
    for mode, r in by_mode.items():
        # Print a one-line summary of every tracked metric for each mode.
        print(f"    {mode:9s} overall {float(r['F1_overall']):.4f} | PER {float(r['F1_PER']):.4f} | "
              f"ORG {float(r['F1_ORG']):.4f} | LOC {float(r['F1_LOC']):.4f} | malformed {r['parsed']['malformed']}")
        # The mode name is left-aligned in a 9-character field for neat
        # column alignment between the two printed rows (zero-shot /
        # few-shot), followed by all four F1 metrics and the malformed count.
    print("\nPhase 5 gate: PASSED")
    # Reaching here means every check() above succeeded.


if __name__ == "__main__":
    # Only run main() when this file is executed directly as a script.
    main()
    # Run the gate.
