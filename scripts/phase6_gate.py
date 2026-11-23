"""Phase 6 gate check. Uses only the shared code in src/ and committed files.

Gate (docs/01-roadmap.md): the report is done and every number in it is
traceable to a logged run.

This script recomputes each number the report states from its source and
checks that the report says the same thing:
  * every F1 in the tables from results/results.csv
  * the transfer matrix, its spreads and the paired gaps (Phase 4 rows)
  * the LLM rows and their malformed counts
  * error counts from results/error_analysis/ and results/phase5/final_errors.csv
  * the "right in all four systems" counts, rebuilt from the saved predictions
    and the real Hindi gold test split
  * tokenizer statistics from results/error_analysis/token_stats.csv
It also checks that no placeholder is left, that every link and figure in the
report exists, and that the figure script can read its data.

Run from the repo root:  uv run python scripts/phase6_gate.py
"""
# Module docstring: Phase 6 produces the project's final written report,
# and this is the capstone gate — rather than checking any one experiment,
# it verifies EVERY individual number that appears in the final report's
# prose and tables is independently reproducible from the project's logged
# source data (results.csv, saved predictions, error-analysis CSVs), so the
# report can't silently drift from or misstate the underlying evidence.

from __future__ import annotations
# Enables postponed evaluation of type hints for modern generic annotations.

import csv
# Standard library csv module: reads results.csv and the various
# error-analysis CSV files.
import gzip
# Standard library gzip module: reads the compressed saved-predictions file
# (predictions.jsonl.gz) without needing to decompress it to disk first.
import json
# Standard library json module: parses each line of the line-delimited JSON
# predictions file.
import os
# Standard library os module: path joins/resolution and existence checks.
import re
# Standard library re module: extracts markdown links from the report text
# to verify they resolve to real files.
import sys
# Standard library sys module: extends the import path for `src` and sibling
# script modules.
from collections import Counter, defaultdict
# Counter: tallies how many gold entities of each type are "wrong in all
# four systems", for the cross-stage error summary.
# defaultdict: groups Phase 4 results rows by transfer-matrix cell, and
# groups saved predictions by system name, without manual key initialization.
from statistics import mean, stdev
# mean: averages F1/gap values across the three random seeds throughout this gate.
# stdev: computes the sample standard deviation across seeds, to report how
# consistent each averaged figure is.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repo root is two directories above this script.
sys.path.insert(0, REPO_ROOT)
# Make the repo root importable so `from src import ...` resolves.
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
# Also add scripts/ itself to the import path, needed to import the sibling
# `phase5_parser` module and (inside main()) `phase6_figures` as plain
# top-level modules.

from seqeval.metrics.sequence_labeling import get_entities  # noqa: E402
# Import seqeval's internal entity-extraction helper directly: given one
# sentence's BIO tag list, get_entities returns the list of (type, start,
# end) spans it contains — used here to compare predicted vs. gold entity
# spans across multiple systems at once, which the simpler evaluate()
# wrapper in src/eval.py doesn't expose directly. `# noqa: E402` suppresses
# the expected "import not at top" warning from the necessary sys.path fixes.

from phase5_parser import format_entities, parse_llm_output, tags_to_entities  # noqa: E402
# Import the Phase 5 parser's entity<->tags conversion functions, used here
# to independently recompute the "parser round-trip ceiling" (how well the
# parser can reconstruct tags even when fed its OWN gold-derived entity
# list, as an upper bound on achievable F1 given its alignment rules).
from src import RESULTS_PATH, evaluate, id2label, load_split  # noqa: E402
# Import the shared results path, F1 evaluator, label-id lookup (used to
# decode saved integer label ids back to tag strings), and the dataset
# split loader — used throughout this gate to reload ground truth directly
# from source rather than trusting any cached/summarized number.

REPORT = os.path.join(REPO_ROOT, "docs", "phase6", "report.md")
# Path to the final Phase 6 report document being audited.
ERR_DIR = os.path.join(REPO_ROOT, "results", "error_analysis")
# Directory holding the various error-analysis CSV/compressed-JSONL files
# produced by earlier notebooks.
FINAL_ERRORS = os.path.join(REPO_ROOT, "results", "phase5", "final_errors.csv")
# Path to a detailed per-error CSV (each row one specific prediction error,
# with its kind, gold/predicted text and type) used for the more granular
# error-pattern claims in the report.
FIGURES = ("phase3_scaling_curve.png", "phase4_transfer_matrix.png", "phase6_hindi_comparison.png", "phase6_error_kinds.png")
# The four figures the final report is expected to embed and reference —
# two produced earlier (Phase 3's scaling curve, Phase 4's transfer matrix
# heatmap) and two produced by this phase's own phase6_figures.py script.
PLACEHOLDERS = ("PENDING", "TODO", "TBD", "XXX")
# Marker strings that must NEVER appear anywhere in a finished report —
# their presence would mean some section was left unfinished/unfilled.
FOUR = ("BiLSTM-CRF", "MuRIL-hi-s42", "IndicBERT", "LLM-few")
# The four specific systems compared in the report's "cross-stage" error
# analysis (which entities does EVERY system get right, or does EVERY
# system get wrong) — matching the shorter system-key naming convention
# used in the saved predictions/error-count files (not the longer "model"
# names used in results.csv).


def check(cond: bool, msg: str) -> None:
    # Shared assertion helper (identical pattern to every other gate script).
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    # Print PASS/FAIL alongside the check's description.
    if not cond:
        # On failure...
        raise SystemExit(f"Gate failed: {msg}")
        # ...halt immediately with a clear message.


def read_csv(path: str) -> list[dict[str, str]]:
    # Small helper reducing repetition: opens a CSV file and returns all its
    # rows as a list of dicts keyed by header column name.
    with open(path, encoding="utf-8", newline="") as f:
        # Open the file as UTF-8 text; newline="" is the standard csv
        # module recommendation.
        return list(csv.DictReader(f))
        # Parse and materialize every row.


def notes_of(row: dict) -> dict[str, str]:
    # Parses a results.csv row's "notes" field into a dict, same
    # implementation used across every phase gate script.
    return dict(p.split("=", 1) for p in row["notes"].split("; ") if "=" in p)
    # Split on "; ", keep parts containing "=", split each on the first "="
    # only, build the dict.


def f4(x: float) -> str:
    # Formats a float to exactly 4 decimal places — the project-wide
    # convention for quoting F1 scores in prose, used throughout this gate
    # to build the exact substrings expected to appear in the report text.
    return f"{x:.4f}"
    # Standard Python format-spec float formatting.


def main() -> None:
    # Entry point running every Phase 6 gate check in sequence.
    check(os.path.exists(REPORT), f"{os.path.relpath(REPORT, REPO_ROOT)} exists")
    # Confirm the report file exists before reading it.
    with open(REPORT, encoding="utf-8") as f:
        # Open the report as UTF-8 text.
        raw = f.read()
        # Read its full raw markdown content.
    text = raw.replace("*", "").replace("`", "").replace("−", "-")   # compare without bold, code marks or the minus sign
    # Build a "normalized" comparison copy of the report text: strip out
    # markdown bold asterisks and inline-code backticks (so a number
    # written as "**0.8274**" or "`0.8274`" still matches a plain
    # "0.8274" search), and replace the Unicode minus sign (−, U+2212,
    # sometimes used by typesetting/export tools) with the plain ASCII
    # hyphen-minus (-), so negative numbers formatted either way still
    # match the ASCII-minus strings this script builds (e.g. via Python's
    # `{:+.4f}` formatting, which always uses ASCII "-").

    def says(s: str, what: str) -> None:
        # A local helper specific to this gate: checks that a given exact
        # substring `s` (an independently recomputed number/phrase) appears
        # somewhere in the normalized report text, labeled by a short
        # description `what` of which claim is being verified.
        check(s in text, f"report states {what}: {s}")
        # Delegate to the shared check() helper, building a message that
        # shows both what was being checked for and the exact string sought.

    print("1. Report hygiene")
    # Section header: basic structural/cleanliness checks on the report
    # file itself, before diving into verifying specific numbers.
    check(not any(p in raw for p in PLACEHOLDERS), f"no placeholder left ({', '.join(PLACEHOLDERS)})")
    # Confirm none of the "unfinished section" marker strings appear
    # anywhere in the RAW (unnormalized) report text.
    check("## Appendix A" in raw, "Appendix A (where each number comes from) is present")
    # Confirm the report includes a dedicated appendix explaining each
    # number's provenance — a project convention for full traceability.
    links = re.findall(r"\]\((?!https?:)([^)#\s]+)", raw)
    # Find every markdown link's URL/path portion: `\]\(` matches the `](`
    # that follows link text, `(?!https?:)` is a negative lookahead
    # excluding links that start with "http:" or "https:" (external links,
    # not checked for local existence), and `([^)#\s]+)` captures the path
    # itself, stopping at a closing paren, a "#" fragment identifier, or whitespace.
    missing = [l for l in links if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(REPORT), l)))]
    # For each extracted relative link path, resolve it relative to the
    # report's OWN directory (docs/phase6/) using os.path.join, normalize
    # away any ".." or "." path segments, and check whether that resolved
    # path actually exists on disk; collect any that don't.
    check(not missing, f"all {len(links)} relative links resolve ({missing or 'none broken'})")
    # Require zero broken relative links; report the total count checked
    # and list any broken ones (or say "none broken" if the list is empty).
    for fig in FIGURES:
        # Check each of the four required figures.
        check(fig in raw and os.path.exists(os.path.join(REPO_ROOT, "results", fig)), f"figure {fig} exists and is in the report")
        # Confirm BOTH that the figure's filename is actually mentioned/
        # embedded somewhere in the report text, AND that the corresponding
        # image file genuinely exists on disk in results/ — catches both "a
        # figure was generated but never referenced" and "the report
        # references a figure that was never actually produced".
    import phase6_figures as pf
    # Import the Phase 6 figure-generation script as a module (deferred to
    # here rather than top-of-file, since it's only needed for this one check).

    pf.load_scores(), pf.load_errors()
    # Call both of phase6_figures.py's data-loading functions directly;
    # since those functions raise SystemExit internally if anything's
    # missing/ambiguous (per their own pick()/validation logic), simply
    # calling them without an exception propagating is itself the proof
    # that the figure script's data sources are intact and correctly shaped.
    check(True, "figure script reads its data from the logged results")
    # Record a (trivially always-true) PASS line documenting that this
    # check was performed — reaching this line means the previous line
    # didn't raise, so the figure script's data loading genuinely succeeded.

    rows = read_csv(RESULTS_PATH)
    # Load every row of the shared results.csv, to be filtered/queried
    # repeatedly throughout the rest of this gate.
    phase = lambda r: notes_of(r).get("phase")
    # A reusable shortcut: extracts a row's "phase" note value (or None if
    # it's not recorded at all), used throughout to distinguish Phase 3
    # rows (no "phase=" note) from Phase 4 rows (explicitly tagged "phase=4").
    early = [r for r in rows if phase(r) is None]
    # Rows with no "phase=" note at all — i.e. rows from Phases 2/3 (which
    # predate this tagging convention), as opposed to Phase 4's explicitly
    # tagged re-use of the same model names.
    num = lambda r, k: float(r[k])
    # A tiny shortcut converting a row's string-valued column `k` to a float.

    print("\n2. Hindi stage scores (results.csv)")
    # Section header: verifies the headline per-model Hindi scores
    # (BiLSTM-CRF, MuRIL handoff, IndicBERT) are each uniquely identified
    # and correctly quoted in the report.
    bil = [r for r in rows if r["model"] == "BiLSTM-CRF"]
    # The single BiLSTM-CRF row (Phase 2 baseline).
    ind = [r for r in rows if r["model"] == "IndicBERT"]
    # The single IndicBERT row (Phase 3).
    hand = [r for r in early if r["model"] == "MuRIL" and r["train_lang"] == r["test_lang"] == "hi" and r["subset_size"] == "300000"
            and notes_of(r)["seed"] == "42"]
    # The specific Phase 3 MuRIL "handoff" row: trained and tested on
    # Hindi, using the 300k-example subset, seed 42, and NOT tagged with a
    # "phase=" note (ensured by filtering `early` rather than all `rows`) —
    # same disambiguation logic as phase6_figures.py's load_scores().
    check(len(bil) == len(ind) == len(hand) == 1, "one BiLSTM-CRF, one IndicBERT and one MuRIL handoff row")
    # Each of the three lookups must find exactly one matching row.
    for name, r in (("BiLSTM-CRF", bil[0]), ("MuRIL handoff", hand[0]), ("IndicBERT", ind[0])):
        # For each of the three models, in this specific display order...
        for k in ("F1_PER", "F1_ORG", "F1_LOC", "F1_overall"):
            # ...check all four of its F1 score columns.
            says(f4(num(r, k)), f"{name} {k}")
            # Confirm this exact 4-decimal-formatted score appears
            # somewhere in the report text.
    seeds3 = [r for r in early if r["model"] == "MuRIL" and r["train_lang"] == r["test_lang"] == "hi" and r["subset_size"] == "300000"
              and r["epochs"] == "3"]
    # All three Phase 3 MuRIL scaling-curve runs at the chosen 300k subset
    # size and 3 epochs (one per seed), used to compute a seed-averaged figure.
    check(sorted(notes_of(r)["seed"] for r in seeds3) == ["42", "43", "44"], "MuRIL 300k has seeds 42, 43, 44 in Phase 3")
    # Confirm exactly the three expected seeds are present (sorted
    # alphabetically, which for these particular two-digit strings also
    # happens to be numeric order).
    says(f4(mean(num(r, "F1_overall") for r in seeds3)), "Phase 3 three-seed MuRIL mean")
    # Confirm the report quotes the mean overall F1 across those three seeds.
    says(f4(min(num(r, "F1_overall") for r in seeds3)), "lowest MuRIL seed")
    # Confirm the report also quotes the single worst-performing seed's
    # score, presumably to characterize the across-seed spread.

    print("\n3. Cross-lingual transfer (Phase 4 rows)")
    # Section header: verifies the full Phase 4 transfer matrix's reported
    # means, standard deviations, and transfer gaps.
    p4 = [r for r in rows if phase(r) == "4"]
    # All rows explicitly tagged "phase=4".
    check(len(p4) == 15, f"15 Phase 4 rows ({len(p4)} found)")
    # Expect exactly 15 rows: 5 transfer-matrix cells x 3 seeds each
    # (matching phase4_gate.py's own row-count check).
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    # Will map each transfer-matrix cell's short key (e.g. "hi-mr") to a
    # dict of {seed: row}, so individual (cell, seed) rows can be looked up directly.
    for r in p4:
        # Group every Phase 4 row.
        n = notes_of(r)
        # Parse this row's notes (needed for its seed value).
        key = {("hi", "hi"): "hi-hi", ("hi", "mr"): "hi-mr", ("hi", "bn"): "hi-bn", ("mr", "mr"): "mr-mr", ("bn", "bn"): "bn-bn"}[
            (r["train_lang"], r["test_lang"])]
        # Translate this row's (train_lang, test_lang) pair into its short
        # display key via a literal lookup dict — will raise a KeyError
        # (crashing the script) if an unexpected language pair ever
        # appears, which is an intentional implicit sanity check.
        groups[key][n["seed"]] = r
        # Store this row under its (cell key, seed) coordinate.
    check(all(sorted(g) == ["42", "43", "44"] for g in groups.values()) and len(groups) == 5, "five matrix cells, three seeds each")
    # Confirm every one of the 5 expected cells has exactly the 3 expected
    # seeds present, and that there are exactly 5 cells total (no missing
    # or unexpected extra cells).
    for key in groups:
        # For each transfer-matrix cell...
        vals = [num(groups[key][s], "F1_overall") for s in ("42", "43", "44")]
        # Collect its overall F1 score across all three seeds, in a fixed
        # seed order.
        says(f"{f4(mean(vals))} ± {f4(stdev(vals))}", f"mean ± SD of {key}")
        # Confirm the report quotes this cell's seed-mean F1 together with
        # its standard deviation, formatted as "mean ± SD" (the "±" plus/minus sign).
    for tgt, name in (("mr", "Marathi"), ("bn", "Bengali")):
        # For each of the two transfer target languages, with its
        # human-readable name for building check descriptions...
        gaps = {s: num(groups[f"{tgt}-{tgt}"][s], "F1_overall") - num(groups[f"hi-{tgt}"][s], "F1_overall") for s in ("42", "43", "44")}
        # For each seed, compute the transfer gap: (in-language F1 on this
        # target) minus (zero-shot F1 transferring from Hindi to this
        # target) — same computation as phase4_gate.py's informational
        # summary, but here it's independently re-derived and checked
        # against the report's claimed numbers.
        g = list(gaps.values())
        # Extract just the gap values (discarding the seed keys) for
        # statistics computation.
        says(f"{mean(g):+.4f} (SD {stdev(g):.4f})", f"{name} transfer gap")
        # Confirm the report quotes this target's mean gap (with explicit
        # sign via `{:+.4f}`) alongside its standard deviation.
        says(" / ".join(f"{x:+.4f}" for x in g), f"{name} per-seed gaps")
        # Confirm the report ALSO quotes the individual per-seed gap values
        # (not just the mean), joined by " / ", each with an explicit sign.
        org = mean(num(groups[f"{tgt}-{tgt}"][s], "F1_ORG") - num(groups[f"hi-{tgt}"][s], "F1_ORG") for s in ("42", "43", "44"))
        # Compute the same transfer-gap calculation, but specifically for
        # the ORG entity type's F1 (rather than overall F1) — ORG being the
        # project's consistently weakest/most-discussed entity type.
        says(f"{name} {org:+.4f}", f"{name} ORG gap")
        # Confirm the report quotes this target's ORG-specific transfer gap.
    for k in ("F1_PER", "F1_ORG", "F1_LOC"):
        # Check each per-type score for the hi->hi (in-language Hindi)
        # cell specifically.
        says(f4(mean(num(groups["hi-hi"][s], k) for s in ("42", "43", "44"))), f"Phase 4 hi to hi mean {k}")
        # Confirm the report quotes the seed-averaged per-type F1 for the
        # Hindi-trained-and-tested baseline cell of the transfer matrix.

    print("\n4. LLM rows (results.csv)")
    # Section header: verifies the Phase 5 LLM rows (zero-shot, few-shot)
    # are correctly quoted, including their malformed-output counts and the
    # headline comparisons against MuRIL.
    llm = {m: next(r for r in rows if r["model"].startswith(f"LLM-{m} (")) for m in ("zero-shot", "few-shot")}
    # Build a {mode: row} dict by finding, for each mode, the single row
    # whose model name starts with "LLM-<mode> (" — `next(...)` with no
    # default will raise StopIteration (crashing the script) if no such row
    # exists, an intentional implicit existence check.
    for m, r in llm.items():
        # For each of the two LLM rows...
        mal = notes_of(r)["malformed"]
        # Extract its recorded malformed-output count string.
        label = "LLM few-shot (k=8)" if m == "few-shot" else "LLM zero-shot"      # as labelled in the report table
        # Build the exact display label the report is expected to use for
        # this row in its results table — the comment notes this must
        # match the report's own table-column labeling convention exactly
        # (few-shot specifically calls out its k=8 example count, zero-shot doesn't).
        says(f"{label} | {f4(num(r, 'F1_PER'))} | {f4(num(r, 'F1_ORG'))} | {f4(num(r, 'F1_LOC'))} | {f4(num(r, 'F1_overall'))} | {mal} |",
             f"{m} row with its malformed count")
        # Confirm the report contains this EXACT markdown-table-row-shaped
        # string: the label, each of the four F1 scores, and the malformed
        # count, all separated by " | " (markdown table cell separators) —
        # a very strict check that the report's table row matches the
        # logged data cell-for-cell, in the expected column order.
    z, fs, mu = llm["zero-shot"], llm["few-shot"], hand[0]
    # Convenience aliases: zero-shot row, few-shot row, and the MuRIL
    # handoff row (from section 2), for the comparison calculations below.
    says(f"{num(fs, 'F1_overall') - num(z, 'F1_overall'):.4f} F1", "the few-shot gain")
    # Confirm the report quotes the exact improvement (few-shot minus
    # zero-shot overall F1) as a plain formatted number followed by "F1".
    d = {k: num(fs, k) - num(mu, k) for k in ("F1_PER", "F1_ORG", "F1_LOC")}
    # Compute, for each entity type, how much better/worse the few-shot LLM
    # is compared to the fine-tuned MuRIL model.
    says(f"PER ({d['F1_PER']:+.3f})", "few-shot minus MuRIL on PER")
    # Confirm the report quotes the PER-type gap, formatted to 3 decimal
    # places with an explicit sign, in parentheses after the type name.
    says(f"ORG {d['F1_ORG']:+.3f}", "few-shot minus MuRIL on ORG")
    # Confirm the report quotes the ORG-type gap (note: no parentheses in
    # this expected string, unlike PER's — matching whatever exact prose
    # phrasing the report actually uses for each).
    says(f"LOC {d['F1_LOC']:+.3f}", "few-shot minus MuRIL on LOC")
    # Confirm the report quotes the LOC-type gap similarly.

    print("\n5. Error counts")
    # Section header: verifies the various error-count claims (totals per
    # system, specific error-kind comparisons) against the error_counts.csv
    # and final_errors.csv source files.
    ec = {(r["lang"], r["system"]): {k: int(r[k]) for k in ("boundary", "missed", "spurious", "type_confusion", "total")}
          for r in read_csv(os.path.join(ERR_DIR, "error_counts.csv"))}
    # Build a dict keyed by (language, system) pairs, each mapping to a
    # dict of that system's five integer error-count columns — a direct
    # lookup table built fresh from the source CSV.
    says(f"MuRIL makes the fewest errors ({ec[('hi', 'MuRIL-hi-s42')]['total']:,}), then BiLSTM-CRF ({ec[('hi', 'BiLSTM-CRF')]['total']:,}), "
         f"IndicBERT ({ec[('hi', 'IndicBERT')]['total']:,}), LLM few-shot ({ec[('hi', 'LLM-few')]['total']:,}), "
         f"and LLM zero-shot ({ec[('hi', 'LLM-zero')]['total']:,})", "the five Hindi error totals")
    # Confirm the report states this exact ranked-by-error-count sentence,
    # naming all five systems' total error counts in the claimed order
    # (lowest to highest), with thousands-separator formatting.
    m, b = ec[("hi", "MuRIL-hi-s42")], ec[("hi", "BiLSTM-CRF")]
    # Convenience aliases for MuRIL's and BiLSTM-CRF's Hindi error-count dicts.
    says(f"({m['missed']} vs {b['missed']} missed)", "MuRIL vs BiLSTM-CRF misses")
    # Confirm the report directly compares these two systems' missed-entity counts.
    says(f"({m['spurious']} vs {b['spurious']})", "MuRIL vs BiLSTM-CRF spurious spans")
    # Confirm the report also compares their spurious-span counts.
    says(f"spurious spans ({m['spurious']} of {m['total']}) and boundaries ({m['boundary']})", "MuRIL's main error kinds")
    # Confirm the report characterizes MuRIL's error PROFILE specifically
    # (its spurious count relative to its own total, plus its boundary count).
    fw, zs = ec[("hi", "LLM-few")], ec[("hi", "LLM-zero")]
    # Convenience aliases for the LLM few-shot and zero-shot error-count dicts.
    says(f"cut spurious spans and misses by {zs['spurious'] - fw['spurious']} each and add {fw['boundary'] - zs['boundary']} boundary errors",
         "few-shot against zero-shot")
    # Confirm the report quotes exactly how much few-shot prompting
    # IMPROVED spurious-span and missed-entity counts compared to
    # zero-shot (zero-shot's count minus few-shot's, i.e. the reduction),
    # and how much it WORSENED boundary-error count (few-shot's minus
    # zero-shot's, i.e. the increase).
    check(zs["spurious"] - fw["spurious"] == zs["missed"] - fw["missed"], "spurious and missed drop by the same amount, as the report says")
    # A genuine data-integrity check (not just a report-text check): verify
    # the claim implicit in the phrase "cut ... by X each" above — that the
    # spurious-span reduction and the missed-entity reduction are actually
    # EQUAL numbers, confirming the "by X each" (shared single number)
    # phrasing used above is factually accurate and not glossing over two different numbers.
    fe = read_csv(FINAL_ERRORS)
    # Load the detailed per-error CSV (one row per individual prediction error).
    typ = lambda r: r["gold_type"] or r["pred_type"]
    # A shortcut: for a given error row, its "entity type" is its
    # gold_type if that's non-empty, otherwise its pred_type (handles
    # "spurious" errors, which have no gold entity at all, so gold_type
    # would be empty and pred_type is the only type information available).
    few = [r for r in fe if r["system"].startswith("LLM few")]
    # Filter to error rows specifically from the LLM few-shot system.
    mur = [r for r in fe if r["system"].startswith("MuRIL")]
    # Filter to error rows specifically from (any seed of) the MuRIL system.
    bd = [r for r in few if r["kind"] == "boundary"]
    # Further filter the few-shot errors down to just "boundary" kind errors.
    longer = sum(len(r["pred"].split()) > len(r["gold"].split()) for r in bd)
    # Count how many of those boundary errors have a predicted span with
    # MORE words than the gold span (i.e. the model's span ran too long),
    # as opposed to too short; `sum(...)` over a generator of booleans
    # counts the True values (Python treats True as 1 in numeric context).
    says(f"{longer} of {len(bd)} longer than gold", "over-long boundary errors")
    # Confirm the report quotes exactly how many of the few-shot model's
    # boundary errors were specifically "too long" ones, out of the total
    # boundary-error count.
    lo = lambda rs: sum(r["kind"] == "type_confusion" and r["gold_type"] == "LOC" and r["pred_type"] == "ORG" for r in rs)
    # A reusable shortcut: given a list of error rows, count how many are
    # specifically a LOCATION entity mistakenly predicted as an
    # ORGANIZATION (a common, specific confusion pattern worth calling out
    # by name in the report).
    says(f"LOC as ORG: {lo(few)} LLM, {lo(mur)} MuRIL", "LOC read as ORG")
    # Confirm the report quotes this specific confusion count for BOTH the
    # LLM and MuRIL systems side by side.
    morg = [r for r in mur if typ(r) == "ORG"]
    # Filter MuRIL's errors down to just ones involving an ORG entity
    # (either as the gold type or, for spurious predictions, the predicted type).
    says(f"ORG alone has {sum(r['kind'] == 'spurious' for r in morg)} spurious spans and {sum(r['kind'] == 'boundary' for r in morg)} boundary errors",
         "MuRIL's ORG errors")
    # Confirm the report quotes, specifically for MuRIL's ORG-related
    # errors, how many are spurious vs. how many are boundary mistakes —
    # drilling into the earlier, coarser "MuRIL's main error kinds" claim
    # with ORG-type-specific detail.
    for tgt in ("mr", "bn"):
        # For each transfer-target language, verify the report's
        # characterization of how MuRIL's error PROFILE shifts when
        # comparing its Hindi-only (zero-shot transfer) checkpoint against
        # its in-language-trained checkpoint, on that target language.
        pair = lambda kind: [ec[(tgt, f"MuRIL-{tgt}-s{s}")][kind] - ec[(tgt, f"MuRIL-hi-s{s}")][kind] for s in (42, 43, 44)]
        # A reusable shortcut (re-defined fresh each loop iteration, so it
        # correctly closes over the current `tgt` value): for a given error
        # kind, compute per-seed the DIFFERENCE between the in-language
        # model's error count and the Hindi-only model's error count on
        # that target language — negative means in-language training
        # REDUCED that error kind, positive means it INCREASED it.
        miss, spur = pair("missed"), pair("spurious")
        # Compute the per-seed differences for both the "missed" and
        # "spurious" error kinds specifically.
        if tgt == "mr":
            # For Marathi specifically...
            check(all(x > 0 for x in miss) and all(x < 0 for x in spur), "Marathi: more misses and fewer spurious spans on every seed")
            # Verify the claimed DIRECTION of the effect holds consistently
            # across ALL THREE seeds (not just on average): in-language
            # training strictly increases missed-entity count and strictly
            # decreases spurious-span count, every single time.
            says(f"Marathi in-language models miss about {round(mean(miss))} more entities than Hindi models and make about "
                 f"{round(abs(mean(spur)))} fewer spurious predictions", "the Marathi error shift")
            # Confirm the report quotes this shift, rounded to whole
            # numbers and averaged across seeds, with "more" and "fewer"
            # phrasing matching the signs just verified above (abs() is
            # used on the spurious mean since it's negative, but the report
            # phrases it as a positive "fewer" count).
        else:
            # For Bengali...
            check(all(x < 0 for x in miss), "Bengali: fewer misses on every seed")
            # Verify a DIFFERENT (weaker) claim for Bengali: only that
            # in-language training reduces missed-entity count on every
            # seed (no equivalent consistent claim is made/checked about
            # spurious spans for Bengali, implying that effect wasn't as
            # clean/consistent for this language).
            says(f"Bengali in-language models miss about {round(abs(mean(miss)))} fewer entities than Hindi models", "the Bengali error shift")
            # Confirm the report quotes Bengali's rounded, averaged
            # reduction in missed entities.

    print("\n6. Cross-stage counts rebuilt from saved predictions and the real gold split")
    # Section header: the most involved re-derivation in this gate — rather
    # than trusting any pre-computed "entities right/wrong in all four
    # systems" count, this section rebuilds that comparison completely from
    # scratch, directly from each system's saved raw per-token predictions
    # and the freshly-reloaded real gold test data.
    gold = [t for _, t in load_split("hi", "test")]
    # Load the real Hindi gold test split fresh, keeping just the tag lists
    # (discarding the word tokens, which aren't needed for this particular comparison).
    preds: dict[str, dict[int, list[str]]] = defaultdict(dict)
    # Will map each system name to a dict of {sentence index: tag list},
    # reconstructed from the saved predictions file.
    with gzip.open(os.path.join(ERR_DIR, "predictions.jsonl.gz"), "rt", encoding="utf-8") as f:
        # Open the gzip-compressed, line-delimited JSON predictions file;
        # "rt" mode reads it as decompressed TEXT (as opposed to "rb" for
        # raw compressed bytes), with explicit UTF-8 encoding for the decompressed text.
        for line in f:
            # Process the file one JSON record (one sentence, from one
            # system) at a time.
            r = json.loads(line)
            # Parse this line's JSON object.
            if r["lang"] == "hi" and r["system"] in FOUR:
                # Only keep records that are both Hindi-language AND from
                # one of the four specific systems this cross-stage
                # comparison cares about (this file may contain many more
                # systems/languages than just these four).
                preds[r["system"]][r["idx"]] = [id2label[int(c)] for c in r["pred"]]   # saved as one label id per token
                # Reconstruct this sentence's predicted tag list: the saved
                # "pred" field holds one INTEGER label id per token (a more
                # compact storage format than full tag strings), so decode
                # each id back to its tag string via id2label, converting
                # each saved value to int first (in case it was stored as a
                # JSON number already, int() is a safe no-op, but guards
                # against it being stored as a numeric string).
    check(all(len(preds[s]) == len(gold) for s in FOUR), "predictions saved for all four systems on all 864 sentences")
    # Confirm every one of the four systems has predictions saved for every
    # single one of the 864 gold sentences (no missing sentences for any system).
    check(all(len(preds[s][i]) == len(g) for s in FOUR for i, g in enumerate(gold)), "one tag per word in every saved prediction")
    # Confirm every single saved prediction has exactly as many tags as its
    # corresponding gold sentence has words — checked exhaustively across
    # all four systems and all 864 sentences.
    all_right = all_wrong = 0
    # Running counters: how many gold entities were correctly predicted by
    # EVERY one of the four systems, and how many were missed by EVERY one
    # of the four systems (chained assignment initializes both to 0).
    wrong_by_type: Counter = Counter()
    # Breaks down the "wrong in all four" count further, by entity type —
    # used to report e.g. how many of those are specifically ORG entities.
    total_by_type: Counter = Counter()
    # Tracks the total count of gold entities seen, broken down by type —
    # used to report the overall gold entity count and give the
    # wrong-by-type numbers context.
    for i, g in enumerate(gold):
        # Iterate every gold sentence, with its index `i` (needed to look
        # up the matching saved predictions, which are keyed by sentence index).
        for t, s, e in get_entities(g):
            # For each gold entity span in this sentence: `get_entities`
            # (from seqeval) returns (type, start_index, end_index) tuples
            # for every entity found in a BIO tag sequence.
            hits = [(t, s, e) in {(tt, ss, ee) for tt, ss, ee in get_entities(preds[sys_][i])} for sys_ in FOUR]
            # For each of the four systems, check whether THIS EXACT gold
            # entity span (t, s, e) — same type, same start, same end — is
            # among the entities that system predicted for this sentence:
            # for each system, extract ITS predicted entity spans for
            # sentence i, put them in a set for fast membership testing,
            # and check whether the gold span is in that set; build a list
            # of four booleans, one per system, in FOUR's order.
            total_by_type[t] += 1
            # Tally this gold entity under its type, contributing to the
            # overall gold-entity-count-by-type tracker.
            all_right += all(hits)
            # If ALL FOUR systems correctly predicted this exact span
            # (`all(hits)` is True), increment the "right in all four" counter.
            if not any(hits):
                # If NONE of the four systems predicted this span at all
                # (every system missed it)...
                all_wrong += 1
                # ...increment the overall "wrong in all four" counter...
                wrong_by_type[t] += 1
                # ...and also tally it under its specific entity type.
    says(f"{all_right:,} are right in all four", "entities right in all four systems")
    # Confirm the report quotes this independently-rebuilt count (with
    # thousands-separator formatting) of entities every single system got right.
    says(f"{all_wrong} are wrong in all four, including {wrong_by_type['ORG']} ORG", "entities wrong in all four systems")
    # Confirm the report quotes the "wrong in all four" count, specifically
    # also calling out how many of those are ORG-type entities (again,
    # highlighting ORG as the consistently hardest entity type).
    says(f"Of {sum(total_by_type.values()):,} gold entities", "the gold entity total")
    # Confirm the report states the correct TOTAL gold entity count (summed
    # across all types), giving the right/wrong-in-all-four counts proper context.

    print("\n7. Tokenizer statistics")
    # Section header: verifies a couple of specific subword-tokenization
    # statistics comparing languages, presumably used in the report to
    # explain WHY certain languages are harder for transformer-based
    # models (more subword fragmentation / more unknown tokens).
    ts = {r["lang"]: r for r in read_csv(os.path.join(ERR_DIR, "token_stats.csv"))}
    # Build a dict mapping language code to its tokenizer-statistics row.
    says(f"{float(ts['bn']['pct_words_with_unk']):.3f}% of Bengali words", "Bengali [UNK] share")
    # Confirm the report quotes the percentage of Bengali words that
    # produced at least one "[UNK]" (unknown/out-of-vocabulary) subword
    # token when tokenized, formatted to 3 decimal places.
    says(f"{float(ts['bn']['mean_subwords']):.3f} vs {float(ts['mr']['mean_subwords']):.3f}", "subwords per word, Bengali vs Marathi")
    # Confirm the report quotes a direct comparison of the average number
    # of subword tokens per word between Bengali and Marathi — presumably
    # supporting a claim about why one language fragments more than the other.

    print("\n8. Parser ceiling")
    # Section header: computes and verifies the Phase 5 parser's
    # theoretical best-case performance — an upper bound showing how much
    # F1 is "lost" purely to the parser's alignment mechanics, independent
    # of any actual model's prediction quality.
    rt = [parse_llm_output(format_entities(tags_to_entities(w, t)), w).tags for w, t in load_split("hi", "test")]
    # For every gold sentence (words `w`, gold tags `t`): first convert the
    # GOLD tags into an entity list (tags_to_entities), serialize that
    # entity list into the exact JSON answer format an LLM would be asked
    # to produce (format_entities), then feed that JSON straight back
    # through the actual parser (parse_llm_output) as if it were a "perfect"
    # model's output — if the parser's alignment rules (R5-R9) were lossless,
    # parsing a gold-derived answer should reconstruct the original gold
    # tags exactly; any discrepancy reveals an inherent ceiling the parser
    # itself imposes (e.g. from its whole-token-alignment requirement).
    says(f4(evaluate(gold, rt)["F1_overall"]), "the parser's round-trip ceiling on gold")
    # Score this round-tripped "rt" tag set against the real gold tags
    # using the shared evaluator, and confirm the report quotes this exact
    # ceiling F1 — if it's below 1.0000, that gap represents the maximum
    # possible F1 any model could achieve given this parser's alignment
    # limitations, even with a flawless understanding of the sentence.

    print("\nPhase 6 gate: PASSED")
    # Reaching here means every check() and says() call above succeeded.


if __name__ == "__main__":
    # Only run main() when this file is executed directly as a script.
    main()
    # Run the gate.
