"""Phase 6 figures, drawn only from logged results (nothing typed in by hand).

  results/phase6_hindi_comparison.png   entity F1 per type, five systems, Hindi gold test
  results/phase6_error_kinds.png        error kinds per system, Hindi gold test

Sources: results/results.csv (scores) and results/error_analysis/error_counts.csv
(error counts, from notebooks/phase5-cross-stage-errors.ipynb). The Phase 3 scaling
curve and the Phase 4 transfer heatmap already exist and are not redrawn here.

Colors are the first slots of the validated categorical palette (surface #fcfcfb):
  3 series, all pairs:      #2a78d6 #eb6834 #1baf7a  (passes; aqua is below 3:1 contrast)
  4 series, adjacent pairs: + #eda100                (passes; aqua and yellow below 3:1)
Because aqua and yellow are below 3:1 on the light surface, the figures carry visible
labels and every number is also in a table in docs/phase6/report.md.

Run from the repo root:  uv run --with matplotlib python scripts/phase6_figures.py
Add --check to only verify that the data needed for the figures is present.
"""
# Module docstring: this script produces the final project report's two
# summary figures, deliberately sourcing every plotted number directly from
# the committed results.csv / error_counts.csv files (never hand-typed), so
# the figures can never silently drift out of sync with the actual logged
# experiment results. It also documents the exact color palette choice and
# its accessibility rationale (contrast ratios against the figure background).

from __future__ import annotations
# Enables postponed evaluation of type hints for modern generic annotations.

import csv
# Standard library csv module: reads both results.csv and error_counts.csv.
import os
# Standard library os module: path joins/resolution for locating input/output files.
import sys
# Standard library sys module: used to check for the "--check" command-line
# flag in sys.argv.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repo root is two directories above this script.
RESULTS = os.path.join(REPO_ROOT, "results", "results.csv")
# Path to the shared results CSV, the source of every model's F1 scores.
ERRORS = os.path.join(REPO_ROOT, "results", "error_analysis", "error_counts.csv")
# Path to the per-system error-kind breakdown CSV, produced earlier by the
# Phase 5 cross-stage-errors notebook.
OUT_COMPARISON = os.path.join(REPO_ROOT, "results", "phase6_hindi_comparison.png")
# Output path for the entity-F1-by-type comparison figure.
OUT_ERRORS = os.path.join(REPO_ROOT, "results", "phase6_error_kinds.png")
# Output path for the error-kinds-by-system figure.

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Four shared hex colors for the figures' chrome: SURFACE is the near-white
# figure/axes background, INK is near-black for primary text (titles,
# labels), INK_2 is a softer dark gray for secondary text (axis ticks,
# captions), and GRID is a light gray for gridlines — all chosen per the
# docstring's accessibility note.
TYPE_COLORS = {"PER": "#2a78d6", "ORG": "#eb6834", "LOC": "#1baf7a"}
# Maps each entity type to its bar color in the comparison figure: blue for
# PERSON, orange for ORGANIZATION, green for LOCATION — one consistent
# color per entity type across the whole figure.
KIND_COLORS = {"boundary": "#2a78d6", "missed": "#eb6834", "spurious": "#1baf7a", "type_confusion": "#eda100"}
# Maps each of the four tracked error kinds to its stacked-bar segment
# color in the error-kinds figure (reusing the same blue/orange/green as
# TYPE_COLORS plus an additional yellow/amber for the fourth category,
# since this figure needs four distinct colors instead of three).
KIND_LABELS = {"boundary": "Boundary", "missed": "Missed", "spurious": "Spurious", "type_confusion": "Type confusion"}
# Human-readable display labels for each error-kind key (the raw CSV column
# names use underscores/lowercase; these are the nicer legend text shown on the figure).


def notes_of(row: dict) -> dict[str, str]:
    # Parses a results.csv row's "notes" field (the "key=value; key=value;
    # ..." format used project-wide) into a dict, same implementation
    # pattern as every phase gate script.
    return dict(p.split("=", 1) for p in row["notes"].split("; ") if "=" in p)
    # Split on "; ", keep only parts containing "=", split each on the first
    # "=" only, build the dict from the pairs.


def load_scores() -> list[dict]:
    """The five Hindi systems, each from one row of results.csv."""
    # Docstring: this function pins down exactly which specific results.csv
    # row represents each of the five systems being compared in the final
    # report (since e.g. "MuRIL" and "LLM-few-shot" could each match
    # multiple logged rows across different phases/seeds/experiments).
    with open(RESULTS, encoding="utf-8", newline="") as f:
        # Open the shared results file for reading.
        rows = list(csv.DictReader(f))
        # Parse every row into a dict keyed by column name.

    def pick(pred, what: str) -> dict:
        # A small local helper: given a predicate function `pred` (applied
        # to each row) and a human-readable description `what` of which
        # system is being looked for, find and return exactly one matching
        # row, failing loudly if zero or more than one match.
        found = [r for r in rows if pred(r)]
        # Filter rows using the given predicate.
        if len(found) != 1:
            # Either no row matched, or ambiguously multiple rows matched.
            raise SystemExit(f"expected exactly one results.csv row for {what}, found {len(found)}")
            # Fail immediately and clearly, naming which system's lookup failed.
            ## WHY reject BOTH zero AND more-than-one matches here, rather
            ## than e.g. silently taking found[0] or found[-1] when there
            ## are multiple: the module docstring's whole premise is that
            ## these figures are drawn ONLY from logged results, "nothing
            ## typed in by hand" — if a predicate like the MuRIL lookup ever
            ## matched two rows (which it genuinely can, since the same
            ## MuRIL checkpoint gets logged again under Phase 4's "phase=4"
            ## tag), picking an arbitrary one would silently make an
            ## editorial choice about which run is "the" result, with no
            ## visible signal that a choice was even made. Failing loudly
            ## instead forces whoever's predicate is ambiguous to tighten it
            ## (as the MuRIL predicate below does, excluding "phase" notes)
            ## rather than letting the figure quietly plot whichever row
            ## Python's list filtering happened to find first.
        return found[0]
        # Exactly one match: return it.

    hi = lambda r: r["train_lang"] == "hi" and r["test_lang"] == "hi"
    # A small reusable predicate: true for any row trained and tested on
    # Hindi — used below to disambiguate the MuRIL lookup (MuRIL appears in
    # results.csv many times across Phases 3 and 4, for different
    # train/test language pairs and seeds).
    picked = [
        # The ordered list of (display name, selected row) pairs for all
        # five systems compared in the Phase 6 report, in the exact order
        # they'll be plotted.
        ## WHY the predicate specifically excludes `"phase" not in
        ## notes_of(r)` on top of model/lang/subset_size/seed: this exact
        ## MuRIL checkpoint (300k, Hindi, seed 42) is logged in results.csv
        ## TWICE — once under its original Phase 3 entry, and again under
        ## Phase 4's hi->hi row (re-evaluating the same checkpoint, tagged
        ## with a "phase=4" note, per phase4_gate.py's REQUIRED pairs).
        ## Without this exclusion, the predicate above it would match BOTH
        ## rows and `pick()` would raise its "found 2, expected 1" error —
        ## this clause is specifically what disambiguates "the original
        ## Phase 3 logging of this run" from "Phase 4's re-use of the same
        ## checkpoint for a different purpose," which otherwise look
        ## identical on every other field being matched.
        ("MuRIL (seed 42)", pick(lambda r: r["model"] == "MuRIL" and hi(r) and r["subset_size"] == "300000"
                                 and notes_of(r).get("seed") == "42" and "phase" not in notes_of(r), "MuRIL 300k seed 42 (Phase 3)")),
        # Pin down the SPECIFIC MuRIL row: model name "MuRIL", trained AND
        # tested on Hindi, using the 300,000-example subset, seed 42
        # specifically (matching the Phase 3 "handoff" run), and explicitly
        # NOT tagged with a "phase=" note (excluding Phase 4's re-use of
        # this same model under a "phase=4" tag) — i.e. the original Phase
        # 3 handoff row, not any later reference to the same checkpoint.
        ("BiLSTM-CRF", pick(lambda r: r["model"] == "BiLSTM-CRF", "BiLSTM-CRF")),
        # The Phase 2 baseline — simple model-name match is unambiguous
        # since there's only ever one BiLSTM-CRF row.
        ("IndicBERT", pick(lambda r: r["model"] == "IndicBERT", "IndicBERT")),
        # The other Phase 3 transformer model — also unambiguous by model name alone.
        ("LLM few-shot", pick(lambda r: r["model"].startswith("LLM-few-shot"), "LLM few-shot")),
        # Phase 5's few-shot-prompted LLM row, matched by its model-name prefix.
        ("LLM zero-shot", pick(lambda r: r["model"].startswith("LLM-zero-shot"), "LLM zero-shot")),
        # Phase 5's zero-shot-prompted LLM row, matched by its model-name prefix.
    ]
    return [{"name": n, **{k: float(r[k]) for k in ("F1_PER", "F1_ORG", "F1_LOC", "F1_overall")}} for n, r in picked]
    # For each (display name, row) pair, build a flat dict combining the
    # display "name" with that row's four F1 score columns converted to
    # floats, using dict-unpacking (`**{...}`) to merge them into one dict
    # per system — this is the final plot-ready data structure returned.


def load_errors() -> list[dict]:
    # Loads each system's error-kind breakdown (how many "boundary",
    # "missed", "spurious", and "type_confusion" errors it made) from the
    # separate error_counts.csv file.
    systems = [("MuRIL (seed 42)", "MuRIL-hi-s42"), ("BiLSTM-CRF", "BiLSTM-CRF"), ("IndicBERT", "IndicBERT"),
               ("LLM few-shot", "LLM-few"), ("LLM zero-shot", "LLM-zero")]
    # The same five systems as load_scores, but paired with the specific
    # "system" key string used in error_counts.csv (which uses shorter,
    # differently-formatted identifiers than results.csv's "model" column,
    # since it's a separate file produced by a different notebook).
    with open(ERRORS, encoding="utf-8", newline="") as f:
        # Open the error-counts CSV for reading.
        rows = {r["system"]: r for r in csv.DictReader(f) if r["lang"] == "hi"}
        # Build a dict mapping each system's key string to its row,
        # filtering to only Hindi-language rows (the file may also contain
        # rows for other languages from broader error-analysis work).
    missing = [s for _, s in systems if s not in rows]
    # Check whether any of the five expected system keys is absent from the
    # loaded Hindi rows.
    if missing:
        # At least one expected system has no matching Hindi row in the file.
        raise SystemExit(f"error_counts.csv has no Hindi row for {missing}")
        # Fail immediately, naming exactly which system key(s) are missing.
    out = []
    # Accumulator for the final per-system error-count dicts.
    for name, key in systems:
        # Process each of the five systems in the fixed display order.
        r = rows[key]
        # Look up this system's error-count row by its CSV key string.
        d = {k: int(r[k]) for k in KIND_COLORS}
        # Build a dict of this system's count for each of the four error
        # kinds (using KIND_COLORS' keys, which conveniently already list
        # exactly those four kind names), converting each to an int.
        assert sum(d.values()) == int(r["total"]), key
        # Sanity-check that the four individual error-kind counts actually
        # sum to the row's separately-recorded "total" column — catches a
        # data-entry or notebook bug where the breakdown and total have
        # drifted apart; `key` is passed as the assert message to identify
        # which system failed, if any.
        out.append({"name": name, **d, "total": int(r["total"])})
        # Append this system's final dict: its display name, its four
        # per-kind counts (unpacked from `d`), and its total count —
        # exactly the shape draw_errors expects.
    return out
    # Return all five systems' error-count dicts, in the fixed display order.


def style(ax):
    # A small shared styling helper applied to both figures' axes, keeping
    # their visual appearance (background, spine visibility, tick color)
    # consistent without duplicating the same matplotlib calls twice.
    ax.set_facecolor(SURFACE)
    # Set the plot area's background to the same off-white SURFACE color as
    # the overall figure background (so there's no visible seam between them).
    for side in ("top", "right"):
        # The top and right plot borders ("spines" in matplotlib terminology).
        ax.spines[side].set_visible(False)
        # Hide them — a minimalist "open" chart style with no box around the
        # plot, just left/bottom axis lines.
    for side in ("left", "bottom"):
        # The remaining left and bottom spines (the actual x/y axis lines).
        ax.spines[side].set_color(GRID)
        # Color them with the soft GRID gray rather than full black, so they
        # recede visually and don't compete with the data itself.
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    # Style the tick marks and their text labels: INK_2 gray color, 9-point
    # font size, and length=0 removes the little tick marks themselves
    # (leaving just the text labels, no protruding lines).


def draw_comparison(scores: list[dict]) -> None:
    # Renders and saves the entity-F1-by-type grouped bar chart comparing
    # all five systems.
    import matplotlib
    # Import matplotlib's top-level package first, deferred to inside this
    # function (rather than at module top-level) so that merely importing
    # phase6_figures.py elsewhere (e.g. for --check mode, or for reuse of
    # its data-loading functions) doesn't require matplotlib to be
    # installed at all.
    ## WHAT BREAKS with a TOP-LEVEL `import matplotlib` instead: the
    ## module docstring explicitly sells `--check` as a fast, data-only
    ## verification mode (`main()` returns before calling draw_comparison/
    ## draw_errors at all when --check is passed). If matplotlib were
    ## imported at module scope, simply running `python scripts/
    ## phase6_figures.py --check` would already require matplotlib
    ## installed just to get past the import statement — defeating the
    ## entire point of offering a lightweight check that doesn't need the
    ## (large, slower-to-install) plotting dependency.
    matplotlib.use("Agg")
    # Select the "Agg" backend explicitly — a non-interactive, headless
    # rendering backend that writes directly to image files without needing
    # a display/GUI, essential for running this script in CI or on a
    # server with no screen.
    import matplotlib.pyplot as plt
    # Import pyplot AFTER setting the backend, since matplotlib backends
    # must be selected before pyplot is imported for the setting to take effect.
    ## WHY this specific order (use() THEN import pyplot) is load-bearing,
    ## not just a style preference: matplotlib locks in and initializes its
    ## rendering backend the first time `matplotlib.pyplot` is imported —
    ## that import is what actually wires up the GUI toolkit or headless
    ## renderer. Calling matplotlib.use("Agg") AFTER pyplot was already
    ## imported either has no effect (the backend is already locked to
    ## whatever pyplot auto-detected, which on a headless CI machine with
    ## no display could raise an error trying to initialize a GUI backend)
    ## or emits a warning, depending on matplotlib version — the ordering
    ## here isn't cosmetic, it's what makes Agg actually take effect.

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200, facecolor=SURFACE)
    # Create a new figure and single Axes: a fixed 8.2x4.4 inch size, high
    # resolution (200 dots per inch, for crisp output in a PDF/report), and
    # the figure's outer background also set to SURFACE (matching the axes
    # background set in style() below, so the whole image is one consistent color).
    style(ax)
    # Apply the shared axis styling helper.
    types = ("PER", "ORG", "LOC")
    # The three entity types to plot as grouped bars, in display order.
    width, gap = 0.25, 0.012
    # `width`: how wide each individual bar is (in x-axis data units, where
    # each system occupies one unit of space). `gap`: a small spacing
    # between adjacent bars within the same system's group, so the three
    # type-bars for one system don't touch.
    for j, t in enumerate(types):
        # For each entity type (with its position j = 0, 1, 2 in the group)...
        xs = [i + (j - 1) * (width + gap) for i in range(len(scores))]
        # Compute this type's bar x-position for every system: `i` is the
        # system's base position (0, 1, 2, 3, 4), and `(j - 1) * (width +
        # gap)` offsets the bar left/center/right within that system's
        # group — j=0 (PER) shifts left by one (width+gap) unit, j=1 (ORG)
        # stays centered (offset 0), j=2 (LOC) shifts right by one unit —
        # creating three side-by-side bars per system, centered on each
        # system's integer x-position.
        ys = [s[f"F1_{t}"] for s in scores]
        # Pull out this entity type's F1 score for every system, in the
        # same order as `xs`.
        ax.bar(xs, ys, width=width, color=TYPE_COLORS[t], edgecolor=SURFACE, linewidth=1.2, label=t, zorder=3)
        # Draw this type's bars: colored per TYPE_COLORS, with a thin
        # SURFACE-colored edge (creates a subtle visual gap/outline between
        # adjacent bars of different colors), labeled with the type name
        # for the legend, and zorder=3 draws bars above the background
        # gridlines (added below at a lower zorder).
    ax.set_ylim(0, 1.0)
    # Fix the y-axis range to the full valid F1 range [0, 1], so bar
    # heights are always visually comparable across different runs of this script.
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Explicit, evenly-spaced tick marks at 0.2 intervals.
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    # Add horizontal gridlines at each y-tick, in the soft GRID color, with
    # zorder=0 so they render BEHIND the bars (drawn at zorder=3 above).
    ax.set_axisbelow(True)
    # Explicitly force gridlines behind all plot elements (belt-and-braces
    # alongside the zorder values, since matplotlib's default axis-below
    # behavior can vary).
    ax.set_xticks(range(len(scores)))
    # One x-tick per system, at integer positions 0 through 4.
    ax.set_xticklabels([f"{s['name']}\noverall {s['F1_overall']:.4f}" for s in scores], color=INK, fontsize=9)
    # Label each system's tick with a two-line label: the system's display
    # name on the first line, and its overall F1 score on the second line —
    # putting the headline overall-F1 number directly on the x-axis for
    # quick reading without needing a separate legend lookup.
    ax.set_ylabel("Entity F1 (seqeval micro)", color=INK_2, fontsize=9)
    # Label the y-axis, clarifying this is entity-level (not token-level)
    # micro-averaged F1, computed via seqeval.
    ax.set_title("Hindi gold test (864 sentences): entity F1 by type", loc="left", color=INK, fontsize=11, pad=24)
    # A left-aligned (rather than centered) title stating exactly what
    # dataset/split this figure covers, with extra top padding (pad=24) to
    # leave room for the legend, which is placed just above the plot area
    # (see bbox_to_anchor below).
    # direct labels on the weakest type, the one the analysis keeps returning to
    for i, s in enumerate(scores):
        # For every system, add a small text label directly above its ORG bar.
        ax.text(i, s["F1_ORG"] + 0.015, f"{s['F1_ORG']:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        # Place the text slightly above the bar's top (y + 0.015, a small
        # visual gap), horizontally centered on the system's x-position,
        # showing the ORG F1 score rounded to 2 decimal places — the
        # comment above explains ORG is specifically called out because
        # it's consistently the weakest-performing entity type discussed
        # throughout the project's analysis, and the module docstring notes
        # this direct labeling compensates for ORG's orange color being
        # below the ideal contrast ratio against the light SURFACE background.
        ## WHY add a text label rather than simply picking a different,
        ## higher-contrast color for ORG specifically: TYPE_COLORS assigns
        ## ONE consistent color per entity type (blue=PER, orange=ORG,
        ## green=LOC), and that mapping is presumably reused across every
        ## figure in the report (the module docstring frames the palette
        ## choice project-wide, not per-figure). Changing ORG's color just
        ## for this chart would break that cross-figure color consistency —
        ## a reader flipping between figures would no longer be able to
        ## rely on "orange always means ORG." Adding a redundant text label
        ## instead fixes the accessibility problem (the number is readable
        ## regardless of color contrast) without sacrificing that
        ## consistency.
    leg = ax.legend(ncols=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    # Add a legend with 3 columns (one per entity type, laid out
    # horizontally), no frame/border box (frameon=False, for the same
    # minimalist style as the hidden spines), anchored just above the plot
    # area (loc="lower left" combined with bbox_to_anchor=(0, 1.0) places
    # the legend's lower-left corner at the axes' top-left corner, i.e.
    # directly above the chart).
    for text in leg.get_texts():
        # Iterate the legend's individual text label objects (one per entity type).
        text.set_color(INK_2)
        # Recolor each legend label to the softer INK_2 gray, matching the
        # rest of the chart's secondary-text color scheme (matplotlib
        # defaults legend text to black otherwise).
    fig.text(0.01, 0.005, "ORG values labelled. Source: results/results.csv. Every value is also in the report's tables.",
             fontsize=7.5, color=INK_2)
    # Add a small caption/footnote in the bottom-left corner of the whole
    # figure (figure-relative coordinates, not axes-relative, hence
    # `fig.text` rather than `ax.text`), in a small font, documenting both
    # why ORG specifically is labeled and where the underlying data comes
    # from — supporting the docstring's accessibility note that every value
    # is also available in tabular form.
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    # Automatically adjust subplot spacing to prevent labels/titles from
    # being cut off, while reserving a small strip at the very bottom (the
    # `rect` parameter leaves 2% of the figure height free at the bottom)
    # for the caption text added just above.
    fig.savefig(OUT_COMPARISON, facecolor=SURFACE)
    # Write the finished figure to disk as a PNG at OUT_COMPARISON,
    # explicitly setting the saved file's background color to SURFACE
    # (matplotlib's savefig can otherwise default to a plain white
    # background regardless of the figure's on-screen facecolor).
    plt.close(fig)
    # Explicitly close the figure to free its memory — important in a
    # script that might create multiple figures in one process (as main()
    # below does), since matplotlib otherwise keeps all created figures in
    # memory until the process exits.


def draw_errors(errs: list[dict]) -> None:
    # Renders and saves the horizontal stacked-bar chart showing each
    # system's error-kind breakdown.
    import matplotlib
    # Same deferred-import pattern as draw_comparison, for the same reason
    # (avoid a hard matplotlib dependency for callers that only need the
    # data-loading functions or --check mode).
    matplotlib.use("Agg")
    # Same headless-backend selection as draw_comparison.
    import matplotlib.pyplot as plt
    # Import pyplot after setting the backend.

    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=200, facecolor=SURFACE)
    # Create a new figure sized slightly shorter than the comparison chart
    # (3.8in vs 4.4in tall), since this one only needs to fit 5 horizontal
    # bars rather than grouped vertical bars with long two-line labels.
    style(ax)
    # Apply the shared axis styling helper.
    n = len(errs)
    # Number of systems being plotted (5).
    ys = list(range(n))[::-1]                      # first system on top
    # Build the y-positions for each system's horizontal bar, REVERSED
    # (`[::-1]`) so that the first system in the `errs` list (MuRIL) ends
    # up at the TOP of the chart rather than the bottom — matplotlib's
    # default y-axis grows upward, so without reversing, list order would
    # read bottom-to-top, which reads unnaturally for a ranked/ordered comparison.
    for y, e in zip(ys, errs):
        # Pair each system's data dict `e` with its (already-reversed) y-position.
        left = 0
        # Running horizontal "left edge" cursor for stacking this system's
        # error-kind segments side by side along the x-axis (a manual
        # stacked-bar implementation, placing each segment starting where
        # the previous one ended).
        for kind in KIND_COLORS:
            # Iterate the four error kinds in KIND_COLORS' fixed dict order
            # (boundary, missed, spurious, type_confusion), so every
            # system's bar stacks its segments in the same consistent order.
            ax.barh(y, e[kind], left=left, height=0.55, color=KIND_COLORS[kind], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            # Draw a horizontal bar segment of width e[kind] (this error
            # kind's count for this system), starting at x=left, at height
            # 0.55 (leaving visual gaps between adjacent systems' bars
            # since each occupies a full y=1 unit of space), colored per
            # KIND_COLORS, with a SURFACE-colored edge for visual separation
            # between stacked segments.
            if e[kind] >= 45:                       # label a segment only where the number fits
                # Only add a text label inside this segment if its count is
                # at least 45 — smaller segments aren't visually wide enough
                # for a readable number to fit inside them.
                ax.text(left + e[kind] / 2, y, str(e[kind]), ha="center", va="center", fontsize=8, color="#ffffff"
                        if kind in ("boundary",) else INK, zorder=4)
                # Place the count as text horizontally centered within this
                # segment (left + half its width) and vertically centered on
                # the bar; use white text specifically for the "boundary"
                # segment (since its blue color is dark enough that white
                # text reads better against it) and the standard INK color
                # for every other segment's lighter-colored background;
                # zorder=4 draws the text above the bar segments themselves.
            left += e[kind]
            # Advance the stacking cursor past this segment, so the next
            # error kind's bar starts exactly where this one ended.
        ax.text(left + 14, y, f"{e['total']:,}", ha="left", va="center", fontsize=9, color=INK, fontweight="bold")
        # After all four segments are drawn, print this system's grand
        # total error count just past the end of its full stacked bar (14
        # data-units of horizontal padding), left-aligned, bold, in the
        # primary INK color, with thousands-separator formatting.
    ax.set_yticks(ys)
    # One y-tick per system, at the (reversed) positions computed above.
    ax.set_yticklabels([e["name"] for e in errs], color=INK, fontsize=9)
    # Label each y-tick with that system's display name, in the ORIGINAL
    # (non-reversed) `errs` order — since `ys` and `errs` were zipped
    # together in original order when drawing, but set_yticks/set_yticklabels
    # here are called with `ys` (the position list) and the label list in
    # `errs`' original order, which lines up correctly because `ys` already
    # encodes which position corresponds to which original-order system.
    ax.set_xlim(0, max(e["total"] for e in errs) * 1.12)
    # Set the x-axis range from 0 to 112% of the largest system's total
    # error count, leaving 12% of extra horizontal space so the
    # grand-total text labels (placed just past each bar's end) always fit
    # within the plot area without being clipped.
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    # Add vertical gridlines at each x-tick (this chart's independent axis,
    # since the bars are horizontal), behind the bars.
    ax.set_axisbelow(True)
    # Force gridlines behind other plot elements.
    ax.set_xlabel("Entity errors on the 864 Hindi gold sentences", color=INK_2, fontsize=9)
    # Label the x-axis, stating the exact dataset this error count is drawn from.
    ax.set_title("Errors by kind, fewest to most", loc="left", color=INK, fontsize=11, pad=24)
    # Left-aligned title, with padding reserved for the legend above the chart.
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLORS[k]) for k in KIND_COLORS]
    # Build custom legend "handles": since this chart draws many small bar
    # segments rather than one bar per label (which would otherwise give
    # matplotlib's automatic legend multiple duplicate entries per error
    # kind), manually create one colored square Rectangle proxy per error
    # kind to represent it cleanly in the legend instead.
    leg = ax.legend(handles, [KIND_LABELS[k] for k in KIND_COLORS], ncols=4, frameon=False, loc="lower left",
                    bbox_to_anchor=(0, 1.0), fontsize=9)
    # Build the legend from those custom handles, paired with the
    # human-readable KIND_LABELS text, in 4 columns (one per error kind),
    # using the same frameless, above-the-chart placement style as the
    # comparison figure's legend.
    for text in leg.get_texts():
        # Iterate the legend's text labels.
        text.set_color(INK_2)
        # Recolor them to the secondary INK_2 gray, matching the project's
        # consistent text color scheme.
    fig.text(0.01, 0.005, "Segments under 45 errors are unlabelled. Source: results/error_analysis/error_counts.csv.",
             fontsize=7.5, color=INK_2)
    # Bottom-left caption explaining the segment-labeling threshold and
    # citing the underlying data source file.
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    # Auto-adjust spacing while reserving bottom room for the caption, same as draw_comparison.
    fig.savefig(OUT_ERRORS, facecolor=SURFACE)
    # Save the finished figure to OUT_ERRORS as a PNG with the SURFACE background.
    plt.close(fig)
    # Free the figure's memory.


def main() -> None:
    # Entry point: load both datasets, print a brief summary, and either
    # stop early (in --check mode) or render and save both figures.
    scores, errs = load_scores(), load_errors()
    # Load the five-system F1 scores and five-system error-kind breakdowns.
    print("scores:", [(s["name"], round(s["F1_overall"], 4)) for s in scores])
    # Print a quick summary of each system's name and rounded overall F1,
    # useful for a sanity-check glance without opening the generated image.
    print("errors:", [(e["name"], e["total"]) for e in errs])
    # Print a quick summary of each system's name and total error count.
    if "--check" in sys.argv:
        # If the script was invoked with the "--check" command-line flag...
        print("data for both figures is present")
        # ...just confirm the data loaded successfully (both load_* calls
        # above would have already raised SystemExit if anything was
        # missing/malformed)...
        return
        # ...and exit without actually rendering any images — useful as a
        # fast CI check that doesn't require matplotlib to be installed or
        # spend time rendering.
    draw_comparison(scores)
    # Render and save the entity-F1-by-type comparison figure.
    draw_errors(errs)
    # Render and save the error-kinds-by-system figure.
    print("wrote", os.path.relpath(OUT_COMPARISON, REPO_ROOT), "and", os.path.relpath(OUT_ERRORS, REPO_ROOT))
    # Print confirmation of both output file paths (shown relative to the
    # repo root for readability), so the user knows exactly what was written.


if __name__ == "__main__":
    # Only run main() when this file is executed directly as a script.
    main()
    # Run the figure-generation entry point.
