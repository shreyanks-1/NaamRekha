"""Tests for scripts/phase5_parser.py. No GPU, no model, no network.

Run from the repo root:  uv run python scripts/test_phase5_parser.py
(also collected by pytest if it is installed).
"""
# Module docstring: a lightweight, dependency-free test suite covering every
# rule (R1-R9) documented in phase5_parser.py — each test below targets one
# specific behavior or edge case of the LLM-output-to-IOB2-tags parser.

from __future__ import annotations
# Enables postponed evaluation of type hints for consistency with the rest
# of the codebase (this file's own functions barely use annotations, but it
# matches the project convention).

import json
# Standard library json module: used by the `j()` helper to build
# well-formed JSON test inputs matching what the parser expects from an LLM.
import os
# Standard library os module: used to compute this script's own directory
# for the sys.path fix below.
import sys
# Standard library sys module: used to extend the import path and (at the
# bottom) to reconfigure stdout's encoding for printing Devanagari text.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Insert this script's own directory (scripts/) at the front of the module
# search path, so `from phase5_parser import ...` below works when this
# file is run directly, regardless of the current working directory.

from phase5_parser import (  # noqa: E402
    # Import the three public functions from the module under test;
    # `# noqa: E402` suppresses the "import not at top of file" lint
    # warning caused by the necessary sys.path fix above.
    format_entities,
    # The function that serializes a gold entity list back into the JSON
    # answer format, used here to test a full round-trip.
    parse_llm_output,
    # The main function under test: maps raw LLM text output onto IOB2 tags.
    tags_to_entities,
    # The inverse-direction helper: converts gold IOB2 tags into an entity
    # list, used both standalone and as part of the round-trip test.
)

W = ["राम", "दिल्ली", "में", "गूगल", "के", "साथ", "काम", "करता", "है"]
# A shared example sentence tokenized into words, reused across many tests
# below as a realistic Hindi sentence containing a PERSON ("राम"/Ram), a
# LOCATION ("दिल्ली"/Delhi), and an ORGANIZATION ("गूगल"/Google), roughly
# meaning "Ram works with Google in Delhi".


def j(*pairs):
    # Small test-only helper that builds the exact JSON text an LLM is
    # expected to produce, from a convenient Python-tuple shorthand.
    return json.dumps([{"text": t, "type": y} for t, y in pairs], ensure_ascii=False)
    # Accepts any number of (text, type) tuples as positional args, builds
    # the corresponding list of {"text": ..., "type": ...} dicts, and
    # serializes that list to a JSON string; ensure_ascii=False keeps the
    # Devanagari characters literal rather than \u-escaped, matching what a
    # real model's raw output would look like.


def test_clean_list():
    # Covers the baseline happy path: a well-formed JSON array naming three
    # entities, none of them overlapping or repeated.
    r = parse_llm_output(j(("राम", "PER"), ("दिल्ली", "LOC"), ("गूगल", "ORG")), W)
    # Parse a clean JSON list tagging "राम" as PER, "दिल्ली" as LOC, and
    # "गूगल" as ORG, against the shared example sentence W.
    assert r.tags == ["B-PER", "B-LOC", "O", "B-ORG", "O", "O", "O", "O", "O"]
    # Each single-word entity should map to exactly one "B-" tag at its
    # token position, with every other token "O" — confirms basic single-
    # token span placement works.
    assert not r.malformed and not r.noisy and r.n_items == 3
    # The output was well-formed (not malformed), didn't need any lenient
    # extraction (not noisy), and all three listed items were valid.


def test_entities_wrapper_object():
    # Covers the `_items()` fallback shape: a dict wrapping the array under
    # an "entities" key, instead of a bare array (part of R1's "wrong
    # shape" tolerance).
    r = parse_llm_output('{"entities": [{"text": "राम", "type": "PER"}]}', W)
    # Feed a JSON OBJECT (not array) whose "entities" key holds the actual list.
    assert r.tags[0] == "B-PER" and not r.malformed
    # The entity should still be found and tagged, and this shape should
    # NOT be considered malformed.


def test_single_bare_entity_object():
    # Covers another `_items()` fallback shape: a single entity object with
    # no wrapping list at all (the model "forgot" the list brackets).
    r = parse_llm_output('{"text": "राम", "type": "PER"}', W)
    # Feed a single bare {"text": ..., "type": ...} object directly.
    assert r.tags[0] == "B-PER" and not r.malformed
    # It should be treated as a one-element entity list and tagged normally.


def test_markdown_fence():
    # Covers R2: a model wrapping its JSON answer in a markdown code fence
    # with an explicit "json" language tag.
    r = parse_llm_output("```json\n" + j(("राम", "PER")) + "\n```", W)
    # Build text that looks like: ```json\n[{"text": "राम", "type": "PER"}]\n```
    assert r.tags[0] == "B-PER" and not r.malformed and r.noisy
    # The entity should still be extracted and tagged correctly, it's NOT
    # malformed, but it IS noisy (R2: formatting drift, tolerated but tracked).


def test_fence_without_language():
    # Same as above but the code fence has no language tag after the
    # opening triple-backtick (just ``` instead of ```json).
    r = parse_llm_output("```\n" + j(("राम", "PER")) + "\n```", W)
    # Build text with a bare, unlabeled code fence around the JSON.
    assert r.tags[0] == "B-PER" and not r.malformed
    # Should still parse correctly regardless of the missing language tag.


def test_prose_around_json():
    # Covers R2's other noisy case: conversational prose surrounding an
    # otherwise valid JSON answer, with no code fence at all.
    r = parse_llm_output("Sure! Here you go:\n" + j(("राम", "PER")) + "\nHope that helps.", W)
    # Simulate a chatty model response that wraps the JSON in a greeting and closing remark.
    assert r.tags[0] == "B-PER" and not r.malformed and r.noisy
    # The entity should still be found (not malformed), and this counts as
    # noisy since the JSON wasn't the entire trimmed response.


def test_empty_list_is_a_valid_answer():
    # Covers the "zero entities" case: a sentence the model correctly
    # determined contains no named entities at all.
    r = parse_llm_output("[]", W)
    # An empty JSON array is a perfectly valid (if unexciting) answer.
    assert r.tags == ["O"] * 9 and not r.malformed and r.n_items == 0
    # All 9 words of W should be tagged "O", this is NOT malformed (it's a
    # legitimate, well-formed empty answer), and zero items were listed.


def test_empty_text_is_malformed():
    # Covers R1's "empty" malformed case.
    r = parse_llm_output("   ", W)
    # Feed text that is entirely whitespace — nothing to parse at all.
    assert r.malformed and r.reason == "empty" and r.tags == ["O"] * 9
    # Must be marked malformed with the specific reason "empty", and every
    # tag falls back to "O" (the safe default) since nothing could be parsed.


def test_prose_only_is_malformed():
    # Covers R1's "no_json" malformed case: there's text, but it contains no
    # JSON brackets at all for the lenient scanner to even attempt.
    r = parse_llm_output("I am unable to find any entities in this sentence.", W)
    # A plain-English refusal/explanation with no JSON syntax whatsoever.
    assert r.malformed and r.reason == "no_json" and r.tags == ["O"] * 9
    # Must be malformed with reason "no_json", tags all fall back to "O".


def test_truncated_json_is_malformed_not_rescued():
    # The first entity is complete, but R1 says broken JSON is an error: no partial credit.
    # Covers R1's "broken_json" case and its explicit "no partial credit"
    # rule — even though the FIRST entity in the array is perfectly
    # well-formed, the array as a whole is truncated mid-second-item, so the
    # whole thing must be rejected rather than salvaging the first entity.
    r = parse_llm_output('[{"text": "राम", "type": "PER"}, {"text": "दिल्', W)
    # Deliberately truncated JSON: a complete first entity followed by the
    # start of a second entity's "text" field that gets cut off.
    assert r.malformed and r.reason == "broken_json" and r.tags == ["O"] * 9
    # Must be malformed with reason "broken_json", and critically, the
    # perfectly valid first entity must NOT be rescued — tags are all "O".


def test_wrong_json_shape_is_malformed():
    # Covers R1's "bad_shape" case across several kinds of validly-parsed
    # JSON that nonetheless aren't a usable entity list/object shape.
    for bad in ('"राम"', "42", '{"result": "ok"}', "true", "null"):
        # Try: a bare JSON string, a bare number, an object with unrelated
        # keys, a boolean, and JSON null — all valid JSON syntactically, but
        # none of them represent an entity list in any recognized shape.
        r = parse_llm_output(bad, W)
        # Parse each one against the shared sentence W.
        assert r.malformed and r.reason == "bad_shape", bad
        # Every single one must be malformed with reason "bad_shape"; the
        # second argument to assert (the `bad` value itself) is used as the
        # failure message, so a failing case identifies exactly which input broke.


def test_unknown_label_skipped_and_counted():
    # Covers R4: entity types that don't map to any known canonical type via
    # LABEL_ALIASES should be skipped (not tagged) but counted separately.
    r = parse_llm_output(j(("राम", "PER"), ("दिल्ली", "CITY"), ("गूगल", "MISC")), W)
    # "CITY" and "MISC" are not in LABEL_ALIASES, unlike "PER" which is canonical.
    assert r.tags[0] == "B-PER" and r.tags[1] == "O" and r.tags[3] == "O"
    # "राम" (valid type) gets tagged; "दिल्ली" (type "CITY") and "गूगल"
    # (type "MISC") are both skipped and remain "O".
    assert r.n_unknown_label == 2 and not r.malformed
    # Exactly two items were skipped for unknown labels, and this does NOT
    # make the whole result malformed — R4 skips are a per-item concern, not
    # a whole-output failure.


def test_label_aliases_and_case():
    # Covers R4's alias mapping and case-insensitivity: lowercase and
    # alternate spellings of entity types should still resolve correctly.
    r = parse_llm_output(j(("राम", "person"), ("दिल्ली", "Location"), ("गूगल", "organisation")), W)
    # "person" (lowercase), "Location" (mixed case), and "organisation"
    # (British spelling, mixed case) should all normalize via
    # `.strip().upper()` plus LABEL_ALIASES lookup.
    assert r.tags[:4] == ["B-PER", "B-LOC", "O", "B-ORG"]
    # All three should resolve to their canonical PER/LOC/ORG tags despite
    # the non-canonical spelling/casing in the input.


def test_invalid_items_counted():
    # Covers R3: several distinct ways a single item can fail the "object
    # with string text and string type" shape check, all counted together.
    r = parse_llm_output('[{"text": "राम"}, "दिल्ली", {"text": 5, "type": "LOC"}, {"text": "  ", "type": "PER"}, '
                         '{"text": "गूगल", "type": "ORG"}]', W)
    # Five items total: (1) missing "type" field entirely, (2) a bare string
    # instead of an object, (3) "text" is a number (5) instead of a string,
    # (4) "text" is only whitespace (squashes to empty), (5) a fully valid item.
    assert r.n_invalid_items == 4 and r.tags[3] == "B-ORG" and not r.malformed
    # The first four items are each invalid for a different reason (all
    # tallied together in n_invalid_items), while the fifth, fully valid
    # item ("गूगल"/ORG, at word index 3) is still correctly tagged — and
    # none of this makes the overall result malformed (R3 failures are
    # per-item, like R4).


def test_span_not_in_sentence_is_unaligned():
    # Covers R8: a listed span whose text simply doesn't appear anywhere in
    # the sentence at all (here, a hallucinated/wrong location).
    r = parse_llm_output(j(("मुंबई", "LOC"), ("राम", "PER")), W)
    # "मुंबई" (Mumbai) isn't anywhere in W (which only mentions "दिल्ली"/Delhi).
    assert r.n_unaligned == 1 and r.tags[0] == "B-PER" and r.tags[1] == "O"
    # Exactly one span (the Mumbai one) is unaligned; "राम" is still
    # correctly tagged since it genuinely exists in the sentence.
    assert r.unaligned_texts == ["मुंबई"]
    # The exact text the model wrote for the failed span is preserved in
    # unaligned_texts for later inspection.


def test_translated_span_is_unaligned():
    # Covers another R8 case: the model gives a semantically correct but
    # translated (different-script) version of a span, which can never
    # align with the original-language sentence tokens.
    r = parse_llm_output(j(("Delhi", "LOC")), W)
    # The model wrote the English translation "Delhi" instead of the
    # original Devanagari "दिल्ली" that actually appears in W.
    assert r.n_unaligned == 1 and r.tags == ["O"] * 9
    # The span can't align (different script entirely), so it's unaligned
    # and every tag stays "O".


def test_partial_word_does_not_match():
    # Covers R5's "whole token" requirement: a substring match inside a
    # single larger token must NOT count as a valid span.
    words = ["दिल्लीवासी", "खुश", "हैं"]
    # "दिल्लीवासी" (meaning "resident of Delhi") is ONE token that happens
    # to contain "दिल्ली" (Delhi) as a substring, but it is not tokenized separately.
    r = parse_llm_output(j(("दिल्ली", "LOC")), words)
    # The model lists just "दिल्ली", which is a substring of the first word
    # but not a whole token on its own.
    assert r.n_unaligned == 1 and r.tags == ["O"] * 3
    # Since "दिल्ली" doesn't align with any WHOLE token's start/end
    # boundaries (R5), it must be rejected as unaligned, not partially
    # matched inside "दिल्लीवासी".


def test_multiword_span():
    # Covers the basic multi-token span case: an entity spanning several
    # consecutive word tokens.
    words = ["अटल", "बिहारी", "वाजपेयी", "आए"]
    # Four tokens: the three parts of a person's full name (Atal Bihari
    # Vajpayee) plus a verb ("आए" = "came").
    r = parse_llm_output(j(("अटल बिहारी वाजपेयी", "PER")), words)
    # The model lists the full three-word name as one PER entity, space-separated.
    assert r.tags == ["B-PER", "I-PER", "I-PER", "O"]
    # The first of the three name tokens gets "B-PER", the next two get
    # "I-PER" (R9's IOB2 scheme), and the unrelated verb token stays "O".


def test_punctuation_spacing_is_forgiven():
    # Naamapadam splits "एम." into "एम" and "."; the model writes it joined (or spaced).
    # Covers R5's specific motivating example: the dataset's tokenizer
    # splits an abbreviation like "एम." (M.) into two separate tokens ("एम"
    # and "."), but a model naturally writes it as one unit, sometimes with
    # a space and sometimes without.
    words = ["एम", ".", "वेंकैया", "नायडू", "ने", "कहा"]
    # Tokenized sentence: "M." (split into "एम" + ".") "Venkaiah Naidu said".
    for text in ("एम. वेंकैया नायडू", "एम . वेंकैया नायडू", "एम.वेंकैया नायडू"):
        # Try three different ways a model might write the same name: with a
        # space after the period (natural writing), with a space both
        # before and after the period (mimicking the token split exactly),
        # and with no spaces at all around the period.
        r = parse_llm_output(j((text, "PER")), words)
        # Parse each variant against the same tokenized sentence.
        assert r.tags == ["B-PER", "I-PER", "I-PER", "I-PER", "O", "O"], text
        # All three variants must align identically to the same four
        # tokens ("एम", ".", "वेंकैया", "नायडू") as one PER span, since
        # squash() strips the whitespace differences between them before
        # comparison; the `text` variable is passed as the assert's failure
        # message to identify which variant broke, if any.


def test_unicode_normalisation_nukta():
    # Covers R5's Unicode-NFC-normalization requirement: a nukta-marked
    # consonant (common in Hindi/Urdu loanwords) can be represented either
    # as one precomposed codepoint or as a base letter plus a combining
    # nukta mark — both must compare equal after squash()'s NFC normalization.
    composed = "ज़"            # ज + nukta
    # The precomposed single-codepoint form of "za" (ज with nukta), stored
    # directly as one Unicode character in this source file.
    words = ["ज़ाकिर", "आए"]
    # The sentence's first token uses whichever form this very source file
    # happens to encode "ज़ाकिर" (Zakir) in — the point of the test is that
    # it shouldn't matter which form either side uses.
    r = parse_llm_output(j(("ज़" + "ाकिर", "PER")), words)   # precomposed ज़ (U+095B) + ाकिर
    # Deliberately construct the model's span text by concatenating the
    # precomposed "ज़" character (U+095B) with the remaining "ाकिर", to
    # ensure a specific codepoint sequence is exercised regardless of how
    # this source file itself might be normalized by an editor.
    assert composed and r.tags == ["B-PER", "O"]
    # `composed and ...` is a style choice ensuring the `composed` variable
    # (defined for documentation purposes above) is referenced so linters
    # don't flag it as unused; the real check is that the span still aligns
    # correctly to "B-PER, O" despite the Unicode representation subtlety.


def test_zero_width_joiner_ignored():
    # Covers R5's invisible-character stripping: a zero-width joiner (ZWJ,
    # used in some Devanagari ligature rendering) embedded in the model's
    # output text must be ignored during alignment.
    words = ["क्रिकेट", "खेलो"]
    # Sentence tokens: "cricket" and "play" (imperative).
    r = parse_llm_output(j(("क्‍रिकेट", "ORG")), words)
    # The model's span text contains a zero-width joiner character
    # (invisible in most renderers) somewhere inside "क्रिकेट" that the
    # actual sentence token does not have.
    assert r.tags == ["B-ORG", "O"]
    # Despite the extra invisible character, squash()'s _INVISIBLE stripping
    # should make the two strings compare equal, so the span still aligns correctly.


def test_repeated_mention_all_occurrences_tagged():
    # Covers R6: when the same name appears multiple times in a sentence,
    # every occurrence must be tagged, not just the first.
    words = ["राम", "ने", "श्याम", "से", "कहा", "कि", "राम", "आएगा"]
    # "Ram said to Shyam that Ram will come" — "राम" (Ram) appears twice, at
    # token index 0 and again at index 6.
    r = parse_llm_output(j(("राम", "PER")), words)
    # The model lists "राम" just once in its output (it's the same named
    # entity, so one mention in the answer is correct LLM behavior).
    assert r.tags == ["B-PER", "O", "O", "O", "O", "O", "B-PER", "O"]
    # BOTH occurrences of "राम" (index 0 and index 6) must be tagged
    # "B-PER", even though the model only listed it once (R6).
    assert r.n_items == 1
    # Only one item was listed by the model — n_items counts listed items,
    # not resulting tagged occurrences.


def test_same_span_listed_twice_counts_once():
    # Covers R7's deduplication: the model might list the identical text
    # twice (redundantly) with the SAME type — this should be treated as
    # one span, not a conflict.
    words = ["राम", "ने", "कहा", "राम"]
    # "Ram said Ram" — again containing "राम" twice.
    r = parse_llm_output(j(("राम", "PER"), ("राम", "PER")), words)
    # The model redundantly lists "राम"/PER twice in its answer.
    assert r.tags == ["B-PER", "O", "O", "B-PER"] and r.n_conflict == 0
    # Both occurrences in the sentence still get tagged (same as R6 above,
    # since it's still the same entity), but since both listed items agree
    # on type, there's no conflict (n_conflict stays 0) — this differs from
    # the next test where the types disagree.


def test_same_span_two_types_first_wins():
    # Covers R7's "first listed wins" tie-breaking rule when the same text
    # is listed with two DIFFERENT, genuinely conflicting types.
    r = parse_llm_output(j(("दिल्ली", "LOC"), ("दिल्ली", "ORG")), W)
    # "दिल्ली" is listed first as LOC, then again as ORG — an inconsistent
    # labeling by the model.
    assert r.tags[1] == "B-LOC" and r.n_conflict == 1
    # The FIRST-listed type (LOC) wins and is what actually gets applied to
    # the token; the conflict is tallied in n_conflict so it can be
    # inspected/reported separately.


def test_nested_longest_first_and_shorter_gets_free_occurrences():
    # Listed shortest first on purpose: the longer span must still win its tokens.
    # Covers R7's "longest first" priority rule, deliberately listing the
    # SHORTER span first in the model's output to prove that listing order
    # doesn't override span-length priority — length always wins regardless
    # of the order spans were listed in.
    words = ["नई", "दिल्ली", "और", "दिल्ली", "में", "मेला"]
    # "New Delhi and Delhi [had] a fair" — "दिल्ली" (Delhi) alone appears at
    # index 1 (as part of "नई दिल्ली"/New Delhi) and again standalone at index 3.
    r = parse_llm_output(j(("दिल्ली", "LOC"), ("नई दिल्ली", "LOC")), words)
    # The model lists the SHORT span "दिल्ली" FIRST, then the LONGER span
    # "नई दिल्ली" (New Delhi) second — testing that processing order is by
    # length, not listing order.
    assert r.tags == ["B-LOC", "I-LOC", "O", "B-LOC", "O", "O"]
    # The longer "नई दिल्ली" span (tokens 0-1) wins at its occurrence despite
    # being listed second, tagged as one B-LOC/I-LOC pair; the shorter
    # "दिल्ली"-only span still gets its OTHER, non-overlapping occurrence at
    # token index 3, tagged as its own B-LOC (R7: "a shorter span only gets
    # the free occurrences").


def test_overlap_with_equal_length_uses_listing_order():
    # Covers R7's tie-breaking by listing order when two spans of EQUAL
    # token length genuinely overlap (share a token).
    words = ["अ", "ब", "स", "कहा"]
    # Four short single-character tokens (used here purely as abstract
    # placeholders to construct a clean overlap scenario) plus "कहा" ("said").
    r = parse_llm_output(j(("अ ब", "PER"), ("ब स", "ORG")), words)
    # Two equal-length (2-token) spans listed: "अ ब" (tokens 0-1) as PER
    # first, then "ब स" (tokens 1-2) as ORG second — they share token index 1.
    assert r.tags == ["B-PER", "I-PER", "O", "O"]
    # Since both spans are the same length, the FIRST-LISTED one ("अ ब"/PER)
    # wins the shared token; the second span ("ब स"/ORG) can't be placed at
    # all since its only occurrence overlaps the already-claimed tokens, so
    # token index 2 stays "O" rather than being half-tagged.
    assert r.n_overlap_dropped == 1
    # The second span ("ब स") is tallied as overlap-dropped since its one
    # and only occurrence was entirely blocked (R8's second case).


def test_adjacent_same_type_spans_do_not_merge():
    # Covers R9: two separate entities of the SAME type that happen to sit
    # right next to each other in the sentence must remain two distinct
    # spans, not merge into one.
    words = ["राम", "श्याम", "आए"]
    # "Ram [and] Shyam came" (informally juxtaposed) — two different PERSON
    # names back-to-back with no separating word, followed by a verb.
    r = parse_llm_output(j(("राम", "PER"), ("श्याम", "PER")), words)
    # Both "राम" and "श्याम" are listed as separate PER entities.
    assert r.tags == ["B-PER", "B-PER", "O"]
    # Critically, the second "राम"-then-"श्याम" token gets its OWN fresh
    # "B-PER" (not "I-PER") — if it were "I-PER", a BIO-tag consumer would
    # incorrectly read the two names as one single two-word entity.


def test_length_always_matches_even_for_garbage():
    # Covers the parser's most basic invariant across EVERY kind of
    # malformed/garbage input: no matter what text comes in, the returned
    # tags list must always have exactly one tag per input word — the
    # pipeline can never crash or return a mismatched-length result.
    garbage = ["", "[", "{", "[[[", "]]]", "null", '{"entities": 3}', "```", "```json", '[{"text":',
               "\u0000", '[{"text": "राम", "type": "PER"}]]]', "{'text': 'राम', 'type': 'PER'}",
               '{"entities": [{"text": "राम", "type": "PER"}]} trailing {"x": 1}']
    # A deliberately wide variety of hostile/broken inputs: an empty
    # string; lone/unbalanced brackets; "null" (valid JSON but wrong
    # shape); an "entities" key holding a non-list value; unlabeled/partial
    # code fences; truncated JSON; a raw null byte; valid JSON with extra
    # trailing closing brackets; Python-style single-quoted "JSON" (invalid
    # JSON syntax); and valid JSON followed by unexpected trailing content.
    for g in garbage:
        # Try every garbage string in turn.
        for words in (W, ["राम"], []):
            # Also vary the sentence itself: the normal 9-word sentence, a
            # single-word sentence, and even an EMPTY word list — checking
            # the length invariant holds regardless of sentence length too,
            # including the edge case of zero words.
            assert len(parse_llm_output(g, words).tags) == len(words), g
            # No matter how broken `g` is, the returned tags list's length
            # must exactly equal the input word list's length; `g` is
            # passed as the assert message to identify which garbage input
            # failed, if any did.


def test_tags_are_valid_iob2_for_any_output():
    # Covers R9 more broadly: no matter what entities are listed (even
    # overlapping or nonsensical multi-word spans), the resulting tag
    # sequence must always be STRUCTURALLY valid IOB2 — every "I-X" tag
    # must be immediately preceded by either a "B-X" or another "I-X" of the
    # exact same type X, never appearing right after "O" or after a
    # different type.
    outs = [j(("राम", "PER"), ("दिल्ली", "LOC"), ("राम दिल्ली", "ORG")),
            j(("में गूगल", "ORG"), ("गूगल के", "LOC"), ("के", "PER")),
            j(("करता है", "ORG"), ("है", "LOC"), ("काम करता है", "PER"))]
    # Three deliberately messy/overlapping sets of listed spans against the
    # shared sentence W, designed to stress-test the overlap-resolution
    # logic (R7) with various overlapping and nested spans of different lengths.
    for o in outs:
        # Check each of the three stress-test outputs.
        tags = parse_llm_output(o, W).tags
        # Parse this output and take just the resulting tag sequence.
        prev = "O"
        # Tracks the previous token's tag as we scan left to right, starting
        # with "O" as the sentinel before the first token.
        for t in tags:
            # Walk the tag sequence token by token.
            if t.startswith("I-"):
                # This token is an "Inside" continuation tag.
                assert prev != "O" and prev[2:] == t[2:], tags
                # For a valid IOB2 sequence, an "I-X" tag must never
                # directly follow "O" (that would be an "orphan" I- with no
                # preceding B-), and the PREVIOUS tag's entity type
                # (`prev[2:]`, stripping the 2-character "B-"/"I-" prefix)
                # must exactly match this tag's type (`t[2:]`) — otherwise
                # two different entities would appear to merge; the full
                # `tags` list is passed as the assert message for easy
                # debugging if this invariant is ever violated.
            prev = t
            # Advance the "previous tag" tracker to this token's tag before
            # moving to the next token.


def test_gold_round_trip():
    # Covers the full pipeline symmetry: converting gold tags to an entity
    # list (tags_to_entities), serializing that list to the prompt-answer
    # JSON format (format_entities), and parsing that JSON back into tags
    # (parse_llm_output) should reproduce the EXACT original gold tags —
    # proving the two directions of the pipeline are true inverses of each
    # other for well-formed, non-overlapping gold data.
    words = ["श्री", "नरेंद्र", "मोदी", "ने", "दिल्ली", "में", "भाजपा", "की", "बैठक", "ली"]
    # "Shri Narendra Modi held a meeting of the BJP in Delhi" — a sentence
    # with a title word ("श्री"/Shri, untagged), a two-word PERSON name, a
    # LOCATION, and an ORGANIZATION.
    gold = ["O", "B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "O", "O", "O"]
    # The corresponding gold IOB2 tags: "श्री" is "O" (a title, not part of
    # the tagged name in this scheme), "नरेंद्र मोदी" is B-PER/I-PER, "दिल्ली"
    # is B-LOC, "भाजपा" is B-ORG, and the remaining function words are "O".
    ents = tags_to_entities(words, gold)
    # Convert the gold tags into the structured entity-list form.
    assert ents == [{"text": "नरेंद्र मोदी", "type": "PER"}, {"text": "दिल्ली", "type": "LOC"},
                    {"text": "भाजपा", "type": "ORG"}]
    # Confirm the reconstructed entity list exactly matches the three
    # expected entities, with the two-word PERSON name correctly joined
    # with a space.
    assert parse_llm_output(format_entities(ents), words).tags == gold
    # Serialize those entities back to the JSON answer format, parse that
    # JSON as if it were a model's output, and confirm the resulting tags
    # are IDENTICAL to the original `gold` tags — the complete round trip.


def test_orphan_i_tag_starts_an_entity():
    # Covers tags_to_entities' documented "orphan I- starts a new entity"
    # behavior (matching how the seqeval scoring library itself treats
    # malformed gold/predicted sequences), as opposed to parse_llm_output's
    # own IOB2 generation (which never produces an orphan I- in the first place).
    ents = tags_to_entities(["क", "ख"], ["I-PER", "I-PER"])
    # Deliberately construct an "invalid" tag sequence where the very first
    # tag is "I-PER" with no preceding "B-PER" — something tags_to_entities
    # must still handle gracefully (e.g. if such malformed gold data ever
    # appeared in the dataset).
    assert ents == [{"text": "क ख", "type": "PER"}]
    # Both orphan "I-PER" tokens get merged into a single PER entity
    # spanning both words — the first "I-PER" triggers starting a new
    # entity (since `cur is None` at that point), and the second "I-PER"
    # then legitimately continues it (same type, cur already open).


def _run_all() -> int:
    # A minimal manual test runner, used so this file can be executed
    # directly with plain `python` (no pytest dependency required), while
    # still being auto-discoverable by pytest if it happens to be installed.
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    # Scan this module's own global namespace for every name that starts
    # with "test_" and is callable (i.e. every test function defined
    # above), sorted alphabetically by name for a deterministic, readable run order.
    failed = 0
    # Counter for how many tests raised an exception (failed their assertions).
    for name, fn in tests:
        # Run each discovered test function in turn.
        try:
            fn()
            # Call the test function; a passing test simply returns
            # normally (an AssertionError means failure).
            print("  PASS ", name)
            # Print a PASS line with the test's name.
        except Exception as e:  # noqa: BLE001
            # Catch ANY exception (not just AssertionError) so one broken
            # test doesn't crash the whole run and prevent later tests from
            # executing; `# noqa: BLE001` suppresses the "blind except"
            # lint warning, since catching broadly is intentional here.
            failed += 1
            # Count this test as failed.
            print("  FAIL ", name, "->", repr(e))
            # Print a FAIL line with the test's name and the exception's repr for diagnosis.
    print(f"\n{len(tests) - failed}/{len(tests)} parser tests passed")
    # Print a final summary line: how many of the total tests passed.
    return failed
    # Return the failure count, used by the caller to decide the process exit code.


if __name__ == "__main__":
    # Only run the test suite when this file is executed directly as a
    # script, not when pytest imports it as a module.
    sys.stdout.reconfigure(encoding="utf-8")
    # Force stdout to use UTF-8 encoding explicitly — necessary on some
    # platforms/terminals (notably Windows) where the default console
    # encoding can't print Devanagari characters, which would otherwise
    # crash the PASS/FAIL print statements above.
    raise SystemExit(1 if _run_all() else 0)
    # Run every test and exit with status code 1 if any failed (truthy
    # failure count) or 0 if all passed (failure count of 0 is falsy) —
    # standard Unix convention so this script can be used in CI/automation
    # and its success/failure detected by the exit code.
