"""Phase 2 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): the BiLSTM-CRF Hindi gold entity F1 is logged in
the results sheet. This script checks that the row exists, follows the
results-sheet rules in docs/03-conventions.md, and matches the output saved
in the Phase 2 notebook, so the number is traceable to a real run.

It does not retrain or re-score the model (the checkpoint lives on Drive).

Run from the repo root:  uv run python scripts/phase2_gate.py
"""
# Module docstring: this gate doesn't recompute anything itself — instead it
# cross-checks that the BiLSTM-CRF result already logged in results.csv is
# consistent, well-formed, and traceable back to the actual notebook run
# that produced it (defense against a stale/hand-edited results row).

from __future__ import annotations
# Enables postponed evaluation of type hints for the modern generic
# annotations (`list[dict[str, str]]`) used in this file.

import csv
# Standard library csv module: used to read the results.csv sheet and to
# re-parse CSV-formatted lines found inside the notebook's saved output.
import json
# Standard library json module: used to parse the notebook (.ipynb) file,
# which is itself stored as JSON.
import os
# Standard library os module: used for path joining/resolution and
# existence checks.
import sys
# Standard library sys module: used to extend the import path so `src` can
# be imported.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Compute the repository root as two directories up from this file
# (scripts/phase2_gate.py -> scripts/ -> repo root), matching the pattern
# used across the other gate scripts.
sys.path.insert(0, REPO_ROOT)
# Prepend the repo root to the module search path so `import src` resolves
# regardless of the working directory this script is invoked from.

from src import COLUMNS, RESULTS_PATH  # noqa: E402
# Import the fixed results-CSV column list and the default results.csv path
# from the shared src package; `# noqa: E402` silences the "import not at
# top of file" lint warning since the sys.path fix must come first.

MODEL = "BiLSTM-CRF"
# The exact model name string expected to appear in the results.csv "model"
# column for Phase 2's run — used to filter the results sheet down to the
# relevant rows.
LANG = "hi"
# The language code (Hindi) this gate checks, since Phase 2 only trains and
# evaluates on Hindi gold data.
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "phase2-bilstm-crf.ipynb")
# Full path to the Phase 2 notebook, whose saved cell outputs are inspected
# to verify the logged result is traceable to an actual run.
PHASE_DOC = os.path.join(REPO_ROOT, "docs", "phase2", "README.md")
# Full path to the Phase 2 write-up doc, which is expected to quote the same
# F1 score that's logged in results.csv.
SCORE_COLS = ["F1_PER", "F1_ORG", "F1_LOC", "F1_overall"]
# The four numeric score columns relevant to this gate's validity checks
# (each must be a sane F1 value in (0, 1]).
# Columns that must agree between results.csv and the notebook's logged row.
# checkpoint_path and notes may be edited after the run (e.g. Drive share link).
TRACE_COLS = ["model", "train_lang", "test_lang", "subset_size", "epochs", *SCORE_COLS]
# Build the full list of columns that must match exactly between the
# results.csv row and the row the notebook printed at training time —
# deliberately excludes checkpoint_path/notes since those can legitimately
# be edited after the fact (e.g. adding a Drive link later) without
# invalidating traceability of the actual scores.
## WHAT BREAKS if checkpoint_path/notes WERE included in TRACE_COLS: the
## entirely legitimate, expected workflow of "train, log the row with
## checkpoint_path='', then later add the real Drive share link once
## uploaded" would make results.csv's checkpoint_path differ from whatever
## the notebook printed at training time (when the link didn't exist yet) —
## the match in section 3 below would then fail every single time for every
## run, even though the actual experiment (model/langs/sizes/scores) is
## provably identical. Limiting TRACE_COLS to only the experiment-defining
## and scoring columns is what makes "editable metadata" and "immutable
## experiment identity" two separate concerns.


def check(cond: bool, msg: str) -> None:
    # Shared assertion helper (same pattern as phase1_gate.py): prints a
    # PASS/FAIL status line and aborts the whole script on first failure.
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    # Print the condition's outcome alongside its human-readable description.
    if not cond:
        # If this particular check failed...
        raise SystemExit(f"Gate failed: {msg}")
        # ...stop the script immediately with a clear failure message,
        # since later checks typically assume earlier ones passed.


## WHY verify by re-parsing the notebook's PRINTED STDOUT OUTPUT rather than,
## say, trusting results.csv on its own, or having the notebook write a
## separate "I really ran" marker file: results.csv is a plain CSV anyone
## could hand-edit to claim any score without ever training anything. A
## notebook's saved cell OUTPUT, on the other hand, is only produced by
## actually executing that cell (log_result's own print statement runs as a
## side effect of logging) — so finding a matching printed line is evidence
## the row traces back to a real execution, not just a claim in a file nobody
## can independently verify. WHAT BREAKS this check: re-running the notebook
## with "clear outputs" and committing it without re-executing would wipe
## this printed evidence and make this check correctly fail.
def notebook_logged_rows() -> list[dict[str, str]]:
    """Rows the notebook printed after log_result (lines starting with MODEL,)."""
    # Docstring: this function reverse-engineers which results rows were
    # actually printed to stdout during the notebook's real execution, by
    # scanning its saved cell outputs for lines that look like CSV rows
    # starting with the model name.
    with open(NOTEBOOK, encoding="utf-8") as f:
        # Open the notebook file as UTF-8 text (it's a JSON document).
        nb = json.load(f)
        # Parse the notebook's JSON structure into a Python dict.
    rows = []
    # Accumulator for every matching logged-row dict found across all cells.
    for cell in nb["cells"]:
        # Iterate every cell (code or markdown) in the notebook's cell list.
        for out in cell.get("outputs", []):
            # Iterate this cell's saved outputs; `.get("outputs", [])`
            # safely handles markdown cells, which have no "outputs" key.
            for line in "".join(out.get("text", [])).splitlines():
                # Jupyter notebook stdout output is stored as a list of text
                # chunks; join them into one string, then split that string
                # back into individual lines to scan line-by-line.
                if line.startswith(MODEL + ","):
                    # A line that starts with "BiLSTM-CRF," looks like a CSV
                    # row for this model (log_result's printed row format),
                    # so treat it as evidence of an actual logged run.
                    rows.append(dict(zip(COLUMNS, next(csv.reader([line])))))
                    # Parse that single line as one CSV row (csv.reader
                    # expects an iterable of lines, hence wrapping `line` in
                    # a list and taking the first parsed row via next()),
                    # then zip the parsed field values together with the
                    # known COLUMNS names to rebuild a {column: value} dict.
    return rows
    # Return every such row found across the whole notebook.


def main() -> None:
    # Entry point running all Phase 2 gate checks in sequence.
    print("1. Results sheet")
    # Section header: checks on the raw results.csv file itself.
    check(os.path.exists(RESULTS_PATH), f"{os.path.relpath(RESULTS_PATH, REPO_ROOT)} exists")
    # Confirm the results.csv file actually exists before trying to read it;
    # the message shows its path relative to the repo root for readability.
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        # Open the results CSV for reading; newline="" is the standard csv
        # module recommendation to avoid platform line-ending quirks.
        reader = csv.DictReader(f)
        # DictReader turns each row into a dict keyed by the header row's
        # column names, inferred automatically from the file's first line.
        header = reader.fieldnames
        # Capture the detected header/column names for validation below.
        rows = list(reader)
        # Materialize every remaining row (as dicts) into a list.
    check(header == COLUMNS, "header matches the fixed columns")
    # Confirm the file's actual header exactly matches the project's fixed
    # COLUMNS schema (same names, same order).

    runs = [r for r in rows if r["model"] == MODEL and r["train_lang"] == LANG and r["test_lang"] == LANG]
    # Filter down to rows that are specifically BiLSTM-CRF runs trained and
    # tested on Hindi (i.e. the exact experiment configuration Phase 2 cares about).
    check(len(runs) >= 1, f"{MODEL} {LANG} -> {LANG} row logged ({len(runs)} found)")
    # Require at least one matching run to exist in the results sheet.
    real = [r for r in runs if "DEBUG" not in r["notes"]]
    # Exclude any rows whose notes field is marked "DEBUG" — these represent
    # throwaway/test runs rather than the real reportable result.
    ## WHY a "DEBUG" substring convention in the free-text `notes` column
    ## instead of a dedicated boolean column (e.g. "is_debug"): COLUMNS is a
    ## fixed, shared schema (src/results.py) used by every phase's logging —
    ## adding a column for one phase's housekeeping need would mean
    ## reworking that shared contract and every existing row in results.csv.
    ## Piggybacking on the already-free-text `notes` field costs nothing
    ## schema-wise. The trade-off: it's a string-matching convention, not an
    ## enforced type, so a typo like "Debug" (wrong case) would silently
    ## fail to exclude a throwaway run.
    check(len(real) >= 1, "at least one non-DEBUG run")
    # Require at least one genuine (non-debug) run to remain after filtering.
    row = max(real, key=lambda r: float(r["F1_overall"]))
    # If multiple genuine runs exist (e.g. from re-running with tweaks),
    # pick the single best one by overall F1 score to treat as "the" result.
    ## WHY max-by-F1_overall rather than, say, the LAST matching row (most
    ## recently appended, since results.csv is append-only): results.csv
    ## accumulates every re-run over time, including earlier attempts with
    ## bugs or worse hyperparameters that were later improved on — "most
    ## recent" is not the same as "best," and could even be a step backward
    ## (e.g. a quick re-run to test something unrelated). Picking the max F1
    ## means this gate (and the doc-matching check in section 4) always
    ## reports and verifies the project's actual best BiLSTM-CRF result,
    ## regardless of what order experiments happened to be logged in.

    print("\n2. Row follows docs/03-conventions.md")
    # Section header: validates the chosen row's values against the
    # project's documented results-sheet conventions.
    scores = {c: float(row[c]) for c in SCORE_COLS}
    # Convert the four score columns from their CSV string form to floats
    # for numeric comparison, keyed by column name.
    for c in SCORE_COLS:
        # Check every one of the four F1 score columns individually.
        check(0.0 < scores[c] <= 1.0, f"{c} = {scores[c]:.4f} is a valid entity F1")
        # An entity-level F1 score must be strictly greater than 0 (a score
        # of exactly 0 would suggest a broken/dummy run, like the all-O
        # baseline from Phase 1) and at most 1.0 (the mathematical maximum
        # for an F1 score) — flags implausible or corrupted values.
        ## WHY strictly `> 0.0` (not `>= 0.0`) specifically: phase1_gate.py's
        ## own sanity check deliberately logs (in a temp file, not the real
        ## results.csv) an all-O dummy baseline that scores exactly 0.0 on
        ## every metric. If this gate only required `>= 0.0`, an accidental
        ## or copy-pasted row from that kind of dummy/placeholder run ending
        ## up in the REAL results.csv would pass as a "valid" BiLSTM-CRF
        ## result. Requiring a strictly positive score encodes "this must be
        ## a model that found at least one entity," ruling out the dummy
        ## baseline shape specifically, not just malformed numbers.
    check(int(row["subset_size"]) > 0, f"subset_size = {int(row['subset_size']):,}")
    # The training subset size must be a positive integer — a real training
    # run must have used some non-zero amount of data.
    check(int(row["epochs"]) > 0, f"epochs = {row['epochs']}")
    # The epoch count must be positive — zero epochs would mean no training happened.
    check(bool(row["checkpoint_path"].strip()), f"checkpoint_path = {row['checkpoint_path']}")
    # The checkpoint_path field, after stripping whitespace, must be
    # non-empty — the trained model weights must actually be saved somewhere.
    check("seed=" in row["notes"], "seed recorded in notes")
    # Convention check: the notes field must mention a recorded random seed
    # (e.g. "seed=42"), ensuring reproducibility of the reported run.

    print("\n3. Traceable to the Phase 2 notebook")
    # Section header: cross-references the results.csv row against what the
    # notebook itself actually printed during execution.
    check(os.path.exists(NOTEBOOK), f"{os.path.relpath(NOTEBOOK, REPO_ROOT)} exists")
    # Confirm the Phase 2 notebook file exists before trying to parse it.
    logged = notebook_logged_rows()
    # Extract every CSV-row-like line the notebook printed to its saved outputs.
    check(len(logged) >= 1, "notebook output contains the logged results row")
    # Require that at least one such logged row was actually found in the
    # notebook's saved output (i.e. the notebook really was run and really
    # did call log_result, rather than results.csv being edited by hand).
    match = any(all(nb_row[c] == row[c] for c in TRACE_COLS) for nb_row in logged)
    # Check whether ANY of the notebook's logged rows matches the
    # results.csv row on every one of the TRACE_COLS fields (model, langs,
    # subset_size, epochs, and all four F1 scores) — a full match on all of
    # these proves the two come from the exact same training run.
    check(match, "results.csv row matches the notebook output (" + ", ".join(TRACE_COLS) + ")")
    # Report pass/fail, listing exactly which columns were compared for transparency.

    print("\n4. Phase 2 doc")
    # Section header: confirms the human-readable Phase 2 write-up actually
    # quotes the same number that's in the results sheet (no silent drift
    # between code-logged numbers and prose claims).
    check(os.path.exists(PHASE_DOC), f"{os.path.relpath(PHASE_DOC, REPO_ROOT)} exists")
    # Confirm the Phase 2 doc file exists.
    with open(PHASE_DOC, encoding="utf-8") as f:
        # Open the doc as UTF-8 text.
        doc = f.read()
        # Read its entire contents into a single string for substring searching.
    check(f"{scores['F1_overall']:.4f}" in doc, f"doc reports the logged F1 {scores['F1_overall']:.4f}")
    # Format the overall F1 score to 4 decimal places (matching how it would
    # typically be quoted in prose) and confirm that exact string appears
    # somewhere in the doc's text — catches the doc quoting a stale/wrong number.

    print(f"\n  {MODEL} Hindi gold F1: overall {scores['F1_overall']:.4f} | "
          f"PER {scores['F1_PER']:.4f} | ORG {scores['F1_ORG']:.4f} | LOC {scores['F1_LOC']:.4f}")
    # Print a human-readable summary line of the model's final scores across
    # all tracked metrics, for quick visual confirmation when running the gate.
    print(f"  Phase 3 bar: MuRIL must clearly beat {scores['F1_overall']:.4f}")
    # Print a forward-looking note: this BiLSTM-CRF score is the baseline
    # that Phase 3's transformer-based model (MuRIL) is expected to exceed.
    print("\nPhase 2 gate: PASSED")
    # Reaching this line means every check() call above succeeded (any
    # failure would have raised SystemExit earlier).


if __name__ == "__main__":
    # Only execute main() when this file is run directly as a script, not
    # when imported as a module.
    main()
    # Run the gate.
