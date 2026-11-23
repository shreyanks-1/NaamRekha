"""Phase 5: turn free-text LLM output into IOB2 tags. Pure Python (no GPU, no torch).

The model is asked for a JSON array like
    [{"text": "नरेंद्र मोदी", "type": "PER"}, {"text": "दिल्ली", "type": "LOC"}]
and `parse_llm_output(text, words)` maps it back onto the sentence's word tokens.

Written rules (applied everywhere, tested in scripts/test_phase5_parser.py)
---------------------------------------------------------------------------
R1  Malformed output is an ERROR, never dropped. Malformed means: empty text,
    no JSON value, broken/truncated JSON, or a JSON value of the wrong shape.
    The sentence is then scored as all `O` and `ParseResult.malformed` is True.
    Only the first `[` or `{` outside a code fence is tried, so truncated output
    is never half-rescued.
R2  Extra text around valid JSON and markdown fences are tolerated; they are
    counted in `noisy` (format drift) but are not malformed.
R3  An item must be an object with a non-empty string "text" and a string
    "type". Anything else is an invalid item: skipped and counted.
R4  Type is upper-cased and mapped through LABEL_ALIASES (PERSON -> PER, ...).
    An unknown type skips the item and is counted in `n_unknown_label`.
R5  Alignment: a span matches a contiguous run of whole word tokens when the
    two strings are equal after Unicode NFC and with all whitespace and
    zero-width joiners removed. (Naamapadam splits punctuation and initials
    into their own tokens, e.g. "एम ." while a model writes "एम.".)
R6  Repeated mentions: EVERY occurrence of a listed span is tagged. A name that
    appears twice in a sentence is the same entity both times.
R7  Overlaps: spans are applied longest first (most tokens), ties by listing
    order. A token that already has a tag is never re-tagged, so a shorter or
    later span only gets the free occurrences. The same text listed twice is
    one span; with two different types the first listed wins (`n_conflict`).
R8  A span that matches nowhere in the sentence (translated, altered or made
    up) is skipped and counted in `n_unaligned`. A span whose every occurrence
    was already taken by R7 is counted in `n_overlap_dropped`.
R9  Tags are valid IOB2: the first token of a span is B-X, the rest I-X, so two
    adjacent spans never merge.
"""
# Module docstring: this file is the "LLM output -> IOB2 tags" bridge for the
# Phase 5 prompting experiments. Instead of a model that natively emits
# per-token tags, an LLM is prompted to list entity mentions as free-text
# JSON, and this module has to robustly, deterministically reconstruct
# token-aligned BIO tags from that JSON, handling every way real LLM output
# can go wrong (malformed JSON, markdown fences, hallucinated spans,
# overlapping/duplicate mentions, unknown entity types) per rules R1-R9
# above, which this file implements line-by-line below.
from __future__ import annotations
# Enables postponed evaluation of type hints for the modern generic
# annotations (`list[str]`, `tuple[int, int, str]`, etc.) used throughout.

import json
# Standard library json module: used both to parse the model's JSON output
# and (via format_entities) to serialize entities back into the exact
# prompt-answer format.
import re
# Standard library re module: used to detect/strip markdown code fences and
# to match/remove invisible whitespace-like characters during alignment.
import unicodedata
# Standard library unicodedata module: used to Unicode-normalize text (NFC)
# before comparing spans, so visually-identical but differently-encoded
# Devanagari/Bengali text still matches.
from dataclasses import dataclass, field
# dataclass: used to define ParseResult as a plain structured result object
# with auto-generated __init__/__repr__.
# field: used to give list-typed dataclass fields a fresh empty list default
# (a bare `= []` default would be shared and mutated across instances).

TYPES = ("PER", "ORG", "LOC")
# The three canonical entity types this project tags, matching the project
# label scheme (minus the "O" tag, which isn't an entity type).

LABEL_ALIASES = {
    # Maps every type spelling/synonym an LLM might plausibly emit to this
    # project's canonical three-letter type code (implements R4).
    "PER": "PER", "PERSON": "PER",
    # Both the canonical short form and the common full word "PERSON" map to "PER".
    "ORG": "ORG", "ORGANIZATION": "ORG", "ORGANISATION": "ORG",
    # Canonical form plus both US and UK spellings of "organization" map to "ORG".
    "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC",
    # Canonical form, the full word "LOCATION", and the common NLP
    # abbreviation "GPE" (geo-political entity) all map to "LOC".
}

_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.S)
# Matches a markdown code fence: three backticks, an optional language tag
# (e.g. "json"), optional whitespace, then the fenced content (captured
# non-greedily in group 1) up to the closing triple-backtick; re.S (DOTALL)
# makes "." match newlines too, since fenced JSON often spans multiple lines.
_INVISIBLE = re.compile(r"[\s​‌‍﻿]+")
# Matches one-or-more characters from: all standard whitespace (\s) plus a
# handful of invisible Unicode characters embedded literally in this
# pattern (zero-width space, zero-width non-joiner, zero-width joiner, and
# a byte-order-mark/zero-width-no-break-space) — used to strip characters
# that are invisible to a human but would otherwise break exact-string
# alignment between model output and the tokenized sentence (R5).


@dataclass
class ParseResult:
    # A structured record of everything learned while parsing one sentence's
    # LLM output: the final tags plus detailed diagnostics about data
    # quality, used both for scoring and for Phase 5's error-analysis notebooks.
    tags: list[str]
    # The final per-token IOB2 tag list, same length as the input `words`
    # list — this is the actual "answer" the rest of the pipeline consumes.
    malformed: bool = False
    # True if R1 applied (the output couldn't be parsed as JSON at all, or
    # had the wrong shape) — in that case `tags` is all "O" and every other
    # diagnostic field below is meaningless/unset.
    reason: str = ""              # why it was malformed ("" if it was not)
    # A short machine-readable string naming which malformed case occurred
    # (e.g. "empty", "no_json", "broken_json", "bad_shape"); empty string
    # when malformed is False.
    noisy: bool = False           # text outside the JSON (prose or fence)
    # True if valid JSON was found but only after skipping surrounding prose
    # or a markdown fence (R2) — tracked separately from "malformed" since
    # noisy-but-parseable output is still scored normally.
    n_items: int = 0              # valid items the model listed
    # Count of entity items that passed the R3/R4 shape and type-alias checks.
    n_invalid_items: int = 0
    # Count of items skipped for failing R3 (not an object, or missing/wrong
    # type for the "text"/"type" fields, or an empty text after squashing).
    n_unknown_label: int = 0
    # Count of items skipped because their "type" string didn't map to any
    # known entity type via LABEL_ALIASES (R4).
    n_unaligned: int = 0
    # Count of listed spans that never matched anywhere in the actual
    # sentence tokens (R8) — e.g. the model paraphrased, translated, or
    # hallucinated a span.
    n_overlap_dropped: int = 0
    # Count of listed spans where every occurrence in the sentence was
    # already claimed by a higher-priority (longer, or earlier-listed) span
    # (R7/R8's second case).
    n_conflict: int = 0
    # Count of cases where the same span text was listed twice with two
    # different types; only the first-listed type is kept (R7).
    unaligned_texts: list[str] = field(default_factory=list)  # the spans R8 skipped, as the model wrote them
    # The raw (not squashed) text of every span counted in n_unaligned, kept
    # for manual error inspection; default_factory=list gives each
    # ParseResult instance its own independent empty list.
    spans: list[tuple[int, int, str]] = field(default_factory=list)  # (first, last, type) word indices
    # The final list of successfully placed entity spans, each as (first
    # word index, last word index, entity type) — a more structured view of
    # the same information encoded in `tags`, convenient for analysis.


def squash(s: str) -> str:
    """Comparison form used for alignment (R5)."""
    # Docstring: documents this is purely a comparison/matching helper, not
    # used for anything user-facing — it normalizes a string so that
    # cosmetic Unicode/whitespace differences don't block span alignment.
    return _INVISIBLE.sub("", unicodedata.normalize("NFC", s))
    # First apply Unicode NFC (Normalization Form C) so that any character
    # that can be represented either as a single precomposed codepoint or as
    # a base character plus combining marks is represented consistently;
    # then strip out every invisible/whitespace character matched by
    # _INVISIBLE, so e.g. "एम ." (with a token-splitting space) and "एम."
    # (written together by a model) squash to the exact same comparison string.


def tags_to_entities(words: list[str], tags: list[str]) -> list[dict[str, str]]:
    """Gold IOB2 -> [{"text", "type"}]. An orphan I- starts a new entity (as seqeval does)."""
    # Docstring: converts gold (ground-truth) BIO tags back into the same
    # {"text", "type"} entity-list shape the LLM is asked to produce — used
    # to build few-shot examples and reference answers for prompting.
    ents: list[dict[str, str]] = []
    # Accumulator list of entity dicts found so far.
    cur = None
    # Tracks the entity dict currently being built (None when not inside an entity).
    for w, t in zip(words, tags):
        # Walk the word tokens and their gold tags in lockstep.
        if t == "O":
            # Outside any entity: whatever entity was being built (if any) is now closed.
            cur = None
            # Reset the "current entity" tracker so the next B-/I- tag starts fresh.
            continue
            # Nothing more to do for an "O" token.
        pre, typ = t.split("-", 1)
        # Split a tag like "B-PER" into its BIO prefix ("B" or "I") and
        # entity type ("PER"), splitting on only the first "-" (maxsplit=1)
        # in case a type name itself ever contained a hyphen.
        if pre == "B" or cur is None or cur["type"] != typ:
            # Start a NEW entity if: this token is explicitly a "B-" (begin)
            # tag, OR there's no entity currently open (cur is None, meaning
            # this "I-" tag has no preceding B- it continues — an "orphan
            # I-", treated leniently as starting a new entity, matching how
            # the seqeval scoring library itself behaves), OR the open
            # entity's type doesn't match this token's type (e.g. an I-ORG
            # immediately following a B-PER, which also can't be a continuation).
            cur = {"text": w, "type": typ}
            # Start a brand-new entity dict with this token as its first word.
            ents.append(cur)
            # Add it to the result list immediately (further words may still
            # extend `cur["text"]` in place via the else-branch below, since
            # dicts are mutable and `ents` holds a reference to the same object).
        else:
            cur["text"] += " " + w
            # This is a genuine continuation ("I-" matching the currently
            # open entity's type): extend the current entity's text with
            # this word, space-separated.
    return ents
    # Return the fully reconstructed list of entity dicts.


def format_entities(ents: list[dict[str, str]]) -> str:
    """The exact answer format the prompt asks for (also used for few-shot answers)."""
    # Docstring: this is the single source of truth for how an entity list
    # is serialized to text — used both when building few-shot prompt
    # examples and (implicitly, by symmetry) when checking the model's
    # output follows the same convention.
    return json.dumps(ents, ensure_ascii=False)
    # Serialize the entity list to a JSON string; ensure_ascii=False keeps
    # Devanagari/Bengali/etc. characters as literal UTF-8 text in the output
    # rather than escaping them as \uXXXX sequences, which is both more
    # readable and matches what a model would naturally produce.


def _load_json(text: str) -> tuple[object, bool]:
    """Return (value, noisy). Raise ValueError(reason) when R1 applies."""
    # Docstring: this is the R1/R2 gatekeeper — it tries progressively more
    # lenient strategies to find a JSON value in the model's raw text,
    # raising a ValueError (caught by the caller) only when every strategy fails.
    t = text.strip()
    # Trim leading/trailing whitespace from the raw model output first.
    if not t:
        # An empty (or all-whitespace) response has no JSON in it at all.
        raise ValueError("empty")
        # R1: empty output is malformed with reason "empty".
    try:
        return json.loads(t), False
        # Happy path (R1's non-malformed case): the ENTIRE trimmed text is
        # valid JSON on its own — nothing noisy about it, so noisy=False.
    except json.JSONDecodeError:
        pass
        # If the whole text isn't valid JSON by itself, fall through to try
        # more lenient extraction strategies below rather than failing immediately.
    for m in _FENCE.finditer(t):
        # Look for every markdown code-fence block in the text (there could
        # be stray formatting around a fenced JSON answer).
        try:
            return json.loads(m.group(1).strip()), True
            # Try parsing just the fenced block's inner content (group 1)
            # as JSON; if it parses, this is noisy=True (R2: content was
            # wrapped in markdown formatting, but still a legitimate answer).
        except json.JSONDecodeError:
            continue
            # This particular fence's content wasn't valid JSON either; try
            # the next fence match, if any.
    body = _FENCE.sub(lambda m: m.group(1), t)          # fence markers removed, content kept
    # No fenced block parsed cleanly on its own, so fall back to a more
    # aggressive strategy: strip every fence's backtick markers (keeping
    # just their inner content in place) across the whole text, producing
    # one combined "body" string to search through — handles output with
    # fences AND extra surrounding prose together.
    starts = [i for i in (body.find("["), body.find("{")) if i != -1]
    # Find the position of the first "[" (start of a JSON array) and first
    # "{" (start of a JSON object) in the body, discarding whichever
    # bracket type doesn't appear at all (str.find returns -1 for "not found").
    if not starts:
        # Neither a "[" nor a "{" appears anywhere in the text.
        raise ValueError("no_json")
        # R1: there's no JSON value to even attempt parsing — malformed with
        # reason "no_json".
    try:
        value, _ = json.JSONDecoder().raw_decode(body[min(starts):])
        # Per the module docstring's R1 rule ("only the first `[` or `{`
        # outside a code fence is tried"), start parsing from whichever
        # bracket comes first in the text (min(starts) is the earlier
        # index); raw_decode parses one JSON value starting at the beginning
        # of the given string and returns (value, index-where-parsing-stopped)
        # — using raw_decode instead of json.loads means trailing garbage
        # after the JSON value doesn't cause a parse failure, and the second
        # return value (end index) is intentionally discarded since we don't
        # need to know how much trailing text followed.
    except json.JSONDecodeError:
        raise ValueError("broken_json") from None
        # If even this lenient attempt fails, the JSON that starts there is
        # genuinely broken or truncated — R1: malformed with reason
        # "broken_json"; `from None` suppresses Python's automatic chained
        # "during handling of the above exception" traceback noise, since
        # the original JSONDecodeError isn't useful to callers.
    return value, True
    # Successfully parsed via the lenient bracket-scanning path — this
    # always counts as noisy=True, since by construction we only reached
    # here after the strict whole-text and fenced-block attempts both failed.


def _items(value: object) -> list[object]:
    """The list of entity items inside a decoded JSON value, or ValueError('bad_shape')."""
    # Docstring: normalizes several plausible JSON shapes a model might
    # produce (a bare list, a dict wrapping a list under an "entities" key,
    # or even a single bare entity object) down to one consistent list of items.
    if isinstance(value, list):
        # The simplest, expected case: the model directly returned a JSON array.
        return value
        # Use it as-is — each element will be validated individually by the caller (R3).
    if isinstance(value, dict):
        # The model might have wrapped the array in an object instead of
        # returning it directly, e.g. {"entities": [...]}.
        for k, v in value.items():
            # Scan every key in the dict looking for one that plausibly
            # holds the entity list.
            if isinstance(k, str) and k.lower() == "entities" and isinstance(v, list):
                # Found a key (case-insensitively) named "entities" whose
                # value is itself a list.
                return v
                # Use that nested list as the items.
        if "text" in value and "type" in value:        # a single bare entity object
            # The model might have returned just ONE entity as a bare object
            # (forgetting to wrap it in a list) when the sentence has
            # exactly one entity — detected by it having both expected keys.
            return [value]
            # Wrap the single object in a one-element list so it's handled
            # uniformly by the rest of the pipeline.
    raise ValueError("bad_shape")
    # The decoded JSON value is neither a list, nor a dict with a
    # recognizable "entities" list, nor a single bare entity object — R1:
    # malformed with reason "bad_shape" (valid JSON, but not usable).


def parse_llm_output(text: str, words: list[str]) -> ParseResult:
    """Map model output onto `words`; always returns one tag per word (R1-R9)."""
    # Docstring: the main entry point tying together JSON extraction (R1/R2),
    # item validation (R3/R4), and token-level span alignment with conflict
    # resolution (R5-R9) into one final ParseResult.
    res = ParseResult(tags=["O"] * len(words))
    # Start with an all-"O" tag list, one per word — this is both the safe
    # default for the malformed case (R1) and the base that successfully
    # aligned spans will overwrite below.
    try:
        value, res.noisy = _load_json(text)
        # Attempt to extract a JSON value from the raw text; on success,
        # also records whether extra formatting/prose was tolerated (R2).
        items = _items(value)
        # Normalize whatever shape the JSON value has into a plain list of
        # candidate entity items.
    except ValueError as e:
        # Either _load_json or _items raised ValueError, meaning R1 applies
        # (truly malformed output in some way).
        res.malformed, res.reason = True, str(e)
        # Record that this result is malformed, with the specific reason
        # string (e.g. "empty", "no_json", "broken_json", "bad_shape") taken
        # from the exception's message.
        return res
        # Return immediately with all-"O" tags — no further processing is
        # attempted on malformed output (R1: "never dropped" but also never
        # partially rescued beyond the lenient JSON-extraction already tried).

    # R3/R4: validate items, keep (squashed text, type, original order)
    wanted: list[tuple[str, str, str]] = []
    # Will hold (squashed comparison text, canonical type, original raw
    # text) for every item that survives shape and type-alias validation.
    for it in items:
        # Validate each candidate entity item individually.
        if not (isinstance(it, dict) and isinstance(it.get("text"), str) and isinstance(it.get("type"), str)):
            # R3: an item must be a JSON object with string-typed "text" and
            # "type" fields; `.get(...)` safely returns None (not a string)
            # if a key is missing, so this also catches missing fields.
            res.n_invalid_items += 1
            # Tally this as an invalid item (R3).
            continue
            # Skip to the next item without adding anything to `wanted`.
        key = squash(it["text"])
        # Compute the alignment-comparison form of this item's text (R5),
        # used both to check for emptiness and later for matching against
        # the sentence's tokens.
        if not key:
            # After squashing (stripping whitespace/invisible chars), the
            # text is empty — e.g. the model wrote only whitespace.
            res.n_invalid_items += 1
            # Also counted as an invalid item under R3 ("non-empty string text").
            continue
            # Skip this item.
        typ = LABEL_ALIASES.get(it["type"].strip().upper())
        # Normalize the type string (trim whitespace, upper-case it) and
        # look it up in LABEL_ALIASES to get the canonical type, or None if unrecognized.
        if typ is None:
            # The type string didn't match any known alias (R4).
            res.n_unknown_label += 1
            # Tally this as an unknown-label skip.
            continue
            # Skip this item; it's never added to `wanted`.
        res.n_items += 1
        # This item passed every R3/R4 check — count it as a valid listed item.
        wanted.append((key, typ, it["text"]))
        # Record it for the alignment stage: squashed key (for matching),
        # canonical type, and the original raw text (kept for diagnostics
        # like unaligned_texts later).

    # R7: dedupe by text (first type wins)
    seen: dict[str, tuple[str, str]] = {}
    # Maps each distinct squashed text to the (type, original raw text) that
    # should "win" for that text — implements "same text listed twice is one
    # span; first-listed type wins" from R7.
    for key, typ, raw in wanted:
        # Walk the valid items in their original listed order (important,
        # since "first listed wins" depends on order).
        if key in seen:
            # This exact span text has already been seen earlier in the list.
            if seen[key][0] != typ:
                # The earlier occurrence had a DIFFERENT type than this one
                # — a genuine type conflict for the same text.
                res.n_conflict += 1
                # Tally the conflict (R7); the earlier (first-listed) type
                # is kept regardless, since we don't overwrite `seen[key]` below.
            continue
            # Whether or not it was a conflict, don't touch `seen[key]`
            # again — the first-listed entry for this text always wins.
        seen[key] = (typ, raw)
        # First time seeing this squashed text: record its type and raw form.

    # token boundaries of the squashed sentence (R5)
    toks = [squash(w) for w in words]
    # Pre-compute the squashed (comparison-normalized) form of every word
    # token in the sentence, in order — this is what listed spans will
    # actually be matched against, so that invisible-character/Unicode
    # differences don't block alignment.
    starts, ends, pos = {}, {}, 0
    # `starts`: maps a character offset in the concatenated squashed
    # sentence to the word-token index that STARTS at that offset.
    # `ends`: maps a character offset to the word-token index that ENDS
    # just before that offset (i.e. the offset right after the token).
    # `pos`: a running character-offset cursor as tokens are concatenated.
    for i, tk in enumerate(toks):
        # Walk each squashed token alongside its index in the word list.
        if tk:
            # Skip tokens that squashed down to an empty string (e.g. a
            # token that was purely whitespace/invisible characters) —
            # they'd otherwise create a zero-width "span" at this position
            # and corrupt the start/end bookkeeping.
            starts.setdefault(pos, i)
            # Record that token index `i` starts at character offset `pos`
            # in the concatenated sentence string — setdefault is used so
            # if two different offsets ever coincided (shouldn't normally
            # happen, but is defensive), the earliest-recorded start wins.
            pos += len(tk)
            # Advance the running offset past this token's squashed length.
            ends[pos] = i
            # Record that token index `i` ends right before the NEW current
            # offset `pos` — i.e. this token occupies squashed-sentence
            # character range [start, pos).
    sent = "".join(toks)
    # Concatenate every squashed token (with NO separator) into one long
    # string — this is the "haystack" that listed entity texts will be
    # searched for as substrings, using the starts/ends maps above to
    # translate character offsets back into token indices.

    def occurrences(key: str) -> list[tuple[int, int]]:
        # Finds every place a given squashed entity text occurs as a
        # whole-token-aligned span in the sentence (R5: must align exactly
        # with token boundaries, not an arbitrary substring mid-token).
        out, i = [], sent.find(key)
        # Start the search at the first occurrence of `key` as a plain
        # substring of `sent` (str.find returns -1 if not found at all).
        while i != -1:
            # Keep searching for more occurrences until find() reports none left.
            if i in starts and (i + len(key)) in ends:
                # Only accept this substring match if its starting offset
                # exactly coincides with a token's start AND its ending
                # offset exactly coincides with a token's end — this is what
                # prevents matching a substring that only partially overlaps
                # a token (e.g. matching inside the middle of a longer word).
                out.append((starts[i], ends[i + len(key)]))
                # Record this occurrence as a (first word index, last word
                # index) pair, using the starts/ends maps to translate the
                # character offsets back into token indices.
            i = sent.find(key, i + 1)
            # Look for the NEXT occurrence, starting the search just past
            # this one's start position (R6: every occurrence must be
            # found, not just the first).
        return out
        # Return every valid whole-token-aligned occurrence found (possibly empty).

    def n_tokens(key: str) -> int:           # token length of the span, 0 if it matches nowhere
        # Computes how many word tokens a span covers, used purely to
        # decide processing order (R7: "longest first").
        occ = occurrences(key)
        # Find where (if anywhere) this key's span occurs.
        return occ[0][1] - occ[0][0] + 1 if occ else 0
        # If it occurs at least once, compute the token count of its FIRST
        # occurrence (last index minus first index, plus one for an
        # inclusive range) — since R6 requires every occurrence of the same
        # text to be the same entity, all occurrences of one key necessarily
        # have the same token length, so checking just the first is enough.
        # If it never occurs, its length is defined as 0 (meaningless for
        # sorting purposes, but it will be dropped by the n_unaligned branch below anyway).

    ordered = sorted(enumerate(seen.items()), key=lambda p: (-n_tokens(p[1][0]), p[0]))
    # Build the processing order required by R7 ("longest first... ties by
    # listing order"): `enumerate(seen.items())` pairs each (key, (type,
    # raw)) item with its original insertion-order index (dicts preserve
    # insertion order in Python); sorting by the key `(-n_tokens(...), p[0])`
    # sorts primarily by NEGATIVE token count (so longer spans — bigger
    # n_tokens — sort first) and secondarily by the original index `p[0]`
    # (so equal-length spans keep their original listed order as a tiebreaker).
    for _, (key, (typ, raw)) in ordered:
        # Process each unique entity span in longest-first, tie-broken-by-order sequence.
        occ = occurrences(key)
        # Re-find all whole-token-aligned occurrences of this span in the sentence.
        if not occ:
            # This span's text doesn't align with any run of whole tokens
            # anywhere in the sentence at all (R8: e.g. the model
            # paraphrased or hallucinated it).
            res.n_unaligned += 1
            # Tally it as unaligned.
            res.unaligned_texts.append(raw)
            # Keep the original (non-squashed) text the model wrote, for
            # later manual inspection of what went wrong.
            continue
            # Nothing more to do for this span.
        placed = False
        # Tracks whether at least one occurrence of this span was
        # successfully tagged (as opposed to all occurrences being blocked
        # by an earlier, higher-priority span — R7/R8).
        for a, b in occ:                                       # R6
            # Try to tag EVERY occurrence of this span (R6: repeated
            # mentions are all the same entity and must all be tagged).
            if all(t == "O" for t in res.tags[a:b + 1]):       # R7
                # Only claim this occurrence's token range if every token in
                # it is still untagged ("O") — since processing happens
                # longest-first, a shorter/later span must never overwrite
                # tokens a longer/earlier span already claimed (R7).
                res.tags[a] = f"B-{typ}"
                # Tag the span's first token as the IOB2 "Begin" tag for
                # this entity type (R9).
                for j in range(a + 1, b + 1):
                    # Tag every remaining token in the span (from just after
                    # the first, through the last, inclusive).
                    res.tags[j] = f"I-{typ}"
                    # Tag it as the IOB2 "Inside" tag for the same entity
                    # type (R9) — ensures adjacent same-type spans never
                    # silently merge into one, since each span gets its own
                    # fresh B- at its start.
                res.spans.append((a, b, typ))
                # Record this successfully placed span in the structured
                # spans list too (for analysis convenience).
                placed = True
                # Mark that at least one occurrence of this span text was placed.
        if not placed:
            # None of this span's occurrences could be placed — every
            # single occurrence's token range was already fully claimed by
            # a higher-priority (longer or earlier) span (R7/R8's second case).
            res.n_overlap_dropped += 1
            # Tally it as dropped-due-to-overlap, distinct from n_unaligned
            # (which means it never matched the sentence text at all).
    res.spans.sort()
    # Sort the recorded spans list by (start, end, type) — i.e. by position
    # in the sentence — since they were appended in longest-first processing
    # order rather than reading order, and reading order is more useful for
    # downstream inspection/reporting.
    return res
    # Return the fully populated ParseResult: final tags plus every
    # diagnostic counter and list accumulated above.
