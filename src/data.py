"""Naamapadam loader and the shared label map."""
# Module docstring: this file is responsible for downloading, extracting, and
# parsing the AI4Bharat "Naamapadam" Indic-language NER dataset, plus defining
# the shared BIO tag vocabulary used throughout the whole project.

from __future__ import annotations
# Enables postponed evaluation of type hints (PEP 563), so annotations like
# `list[Sentence]` work as string-deferred types even on Python versions where
# `list[...]` generic subscripting in annotations would otherwise need care.

import json
# Standard library JSON module: used to parse each line of the dataset files,
# which are stored in JSON-lines format (one JSON object per line).
import os
# Standard library OS module: used for filesystem path joins, existence
# checks, and walking directories to locate extracted dataset files.
import zipfile
# Standard library zipfile module: used to extract the downloaded dataset
# archive (the Naamapadam data ships as a .zip per language).
from collections import Counter
# Counter: a dict subclass for counting hashable objects; used here to tally
# how many times each NER tag appears across a dataset split.

from huggingface_hub import hf_hub_download
# hf_hub_download: downloads (and caches locally) a single file from a
# Hugging Face Hub repository, used to fetch the per-language dataset zip.

REPO_ID = "ai4bharat/naamapadam"
# The Hugging Face Hub dataset repository id that hosts the Naamapadam
# Indic NER corpus; all language zips are fetched from this one repo.

## WHY a fixed list (not a set, not derived dynamically from the data): the
## *order* is the contract. Every model's output layer has one logit per index
## in this list, and every saved checkpoint/confusion matrix/eval script
## assumes index 1 always means "B-PER". If this were instead built by
## scanning the dataset for unique tags (e.g. `sorted(set(all_tags))`), the
## order could differ between runs, languages, or dataset versions whenever
## a tag was missing from one split — silently producing a DIFFERENT id
## mapping per run. A model trained under one mapping and evaluated/loaded
## under another would get coherent-looking but completely wrong predictions
## (e.g. predicting "I-PER" where "B-ORG" was meant) with no error raised.
# Fixed label order; ids are the list indices.
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
# The canonical BIO (Begin/Inside/Outside) tag set for this NER task:
# "O" = outside any entity, B-/I- prefixes mark the start/continuation of a
# PERson, ORGanization, or LOCation span respectively. The list order below
# fixes each tag's integer id, so every model in the project shares one
# consistent label numbering.
label2id = {label: i for i, label in enumerate(LABELS)}
# Builds a dict mapping each tag string to its position (index) in LABELS,
# e.g. {"O": 0, "B-PER": 1, ...} — used when encoding string tags into the
# integer ids a neural network's output layer expects.
id2label = {i: label for i, label in enumerate(LABELS)}
# The inverse mapping of label2id: integer id -> tag string — used when
# decoding a model's predicted integer label ids back into BIO tag strings.

SPLITS = ("train", "val", "test")
# The three valid dataset partitions: a tuple (immutable) since this set of
# split names never changes and is used for membership checks and iteration.

# (tokens, tags), equal length.
Sentence = tuple[list[str], list[str]]
# Type alias documenting the shape of one dataset example: a tuple of
# (list of word tokens, list of BIO tag strings), where both lists are the
# same length (one tag per token).


def _ensure_extracted(lang: str) -> str:
    # Leading underscore marks this as a private helper, not part of the
    # module's public API — it's only used internally by load_split.
    """Download the language zip (cached) and extract it once; return the dir."""
    # Docstring: explains the function downloads-if-needed and caches its
    # extraction so repeated calls don't redo the unzip work.
    zip_path = hf_hub_download(
        # Ask huggingface_hub to fetch (or reuse its local cache of) the
        # dataset archive for the requested language; returns the local path.
        REPO_ID, f"data/{lang}_IndicNER_v1.0.zip", repo_type="dataset"
        # REPO_ID is the dataset repo; the filename is built from `lang`
        # (e.g. "hi" for Hindi) to target that language's specific zip file;
        # repo_type="dataset" tells the Hub this is a dataset repo, not a model.
    )
    out_dir = zip_path + ".extracted"
    # Derive a deterministic extraction directory name by suffixing the zip's
    # cached path with ".extracted", so it sits next to the cached zip file.
    ## WHY derive out_dir from zip_path instead of a fixed path like
    ## "./data/{lang}": hf_hub_download's cache path already encodes the repo,
    ## revision, and filename, so re-using it means different languages (and
    ## different dataset revisions, if the Hub file ever changes) automatically
    ## get their own separate extraction folder with zero extra bookkeeping.
    ## A fixed path would need its own per-language, per-revision naming logic
    ## or different languages would overwrite each other's extracted files.
    if not os.path.isdir(out_dir):
        # Only extract if we haven't already done so for this exact cached
        # zip — avoids redundant disk I/O on repeated calls/runs.
        ## WHAT BREAKS without this existence check: every call to
        ## _ensure_extracted (e.g. once per split per language) would re-run
        ## zf.extractall, redoing multi-GB disk I/O every time load_split is
        ## called instead of once. It would still be *correct* (extractall is
        ## idempotent), just needlessly slow on repeated notebook re-runs.
        with zipfile.ZipFile(zip_path) as zf:
            # Open the downloaded zip archive for reading; the `with` block
            # ensures the file handle is closed automatically afterward.
            zf.extractall(out_dir)
            # Extract every file inside the archive into out_dir, preserving
            # the archive's internal folder structure.
    return out_dir
    # Hand back the path to the (now guaranteed-to-exist) extracted directory
    # so the caller can locate individual split files inside it.


def _find_file(out_dir: str, lang: str, split: str) -> str:
    # Private helper: locates one specific split file inside the extracted
    # directory tree, which may have nested subfolders.
    """Return the path of {lang}_{split}.json under out_dir."""
    # Docstring: documents the expected filename pattern being searched for.
    target = f"{lang}_{split}.json"
    # Build the exact filename we're looking for, e.g. "hi_train.json".
    ## WHY walk the whole directory tree instead of a hardcoded path like
    ## f"{out_dir}/{lang}_{split}.json": the Hub zip's internal folder layout
    ## isn't guaranteed identical across languages/releases (e.g. some zips
    ## may nest files one level deeper). os.walk makes the loader tolerant to
    ## that structural drift at the cost of an O(num_files_in_tree) scan.
    ## WHAT BREAKS with a hardcoded path instead: it would work for whichever
    ## language's layout was used to write the hardcoded path, then raise a
    ## confusing FileNotFoundError for any other language whose zip nests
    ## files differently — a silent-until-you-add-a-new-language bug.
    for root, _, files in os.walk(out_dir):
        # Recursively walk every directory under out_dir; `root` is the
        # current directory path, `_` (unused) would be subdirectory names,
        # and `files` is the list of filenames in that directory.
        if target in files:
            # Check whether the target filename exists in this directory's
            # file listing.
            return os.path.join(root, target)
            # Found it — join the directory and filename into a full path
            # and return immediately.
    raise FileNotFoundError(f"{target} not found under {out_dir}")
    # If the walk completes without finding the file, fail loudly with a
    # clear error identifying exactly what was expected and where.


def load_split(lang: str, split: str) -> list[Sentence]:
    """Return one split as (words, tags) pairs. lang e.g. "hi"/"mr"/"bn",
    split one of SPLITS. Empty sentences dropped; tags are strings."""
    # Docstring: documents the inputs (language code, split name) and the
    # output shape, plus the filtering behavior (dropping empty sentences).
    if split not in SPLITS:
        # Validate the split argument early against the known-good set of
        # split names, rather than letting a typo fail silently later.
        ## WHY validate here instead of just letting _find_file fail: a typo
        ## like split="vall" would otherwise propagate into the filename
        ## f"{lang}_{split}.json" inside _find_file and surface as a generic
        ## FileNotFoundError that doesn't mention "split" at all — forcing
        ## whoever hits it to trace back through two function calls to find
        ## the real mistake. Failing fast here, at the API boundary, turns an
        ## unclear downstream crash into an immediate, specific ValueError.
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        # Raise a clear, actionable error message showing both the valid
        # options and the invalid value the caller actually passed.

    path = _find_file(_ensure_extracted(lang), lang, split)
    # First ensure the language's archive is downloaded and extracted, then
    # locate the specific JSON file for this split within that directory.
    rows: list[Sentence] = []
    # Accumulator list that will hold every (words, tags) sentence pair read
    # from the file.
    with open(path, encoding="utf-8") as f:
        # Open the split file as UTF-8 text (needed for Indic-script
        # characters) and ensure it's closed automatically afterward.
        for line in f:
            # The file is in JSON-Lines format: iterate it line by line
            # rather than loading/parsing it as one big JSON document.
            ## WHY iterate `for line in f` instead of `json.load(f)`: this
            ## file is JSON-Lines (one JSON object per line), not a single
            ## JSON array, so json.load(f) would raise a JSONDecodeError
            ## immediately on hitting the second "{" after the first
            ## object's closing "}". Line-by-line iteration over an open
            ## file is also lazy/streaming — Python doesn't materialize the
            ## whole ~1M-line train file as one giant string before parsing
            ## anything, which matters given hi_train.json's size.
            line = line.strip()
            # Remove leading/trailing whitespace (including the trailing
            # newline) so blank-line and JSON-parsing checks work correctly.
            if not line:  # skip blank lines in the file
                continue
                # An empty line after stripping means there's nothing to
                # parse on this line; skip straight to the next iteration.
            obj = json.loads(line)
            # Parse this line's JSON text into a Python dict, e.g.
            # {"words": [...], "ner": [...]}.
            words, tags = obj["words"], obj["ner"]
            # Pull out the token list and its corresponding NER tag list from
            # the parsed JSON object using the dataset's known field names.
            ## WHY no `assert len(words) == len(tags)` here: this loader is
            ## deliberately fast and trusting — it doesn't re-verify per-row
            ## alignment on every load. That verification is done once,
            ## separately, during dataset inspection (see the phase1 notebook's
            ## mismatch counter). WHAT IT MEANS IN PRACTICE: if a future
            ## dataset revision ever produced a misaligned row, load_split
            ## would NOT raise here — it would silently return a (words, tags)
            ## pair of different lengths, and the failure would only surface
            ## later and more confusingly, e.g. as an off-by-one label when
            ## zipping tokens to tags for training.
            if not words:  # drop zero-length sentences
                continue
                # Skip sentences with no tokens at all — they carry no
                # training/evaluation signal and would just be noise.
                ## WHAT BREAKS without this filter: an empty `words` list
                ## means an empty tag list too, and a tokenizer/collator fed
                ## a zero-length sequence can either error outright (many HF
                ## tokenizers assume at least one token) or, if it doesn't
                ## error, contribute a sequence with zero valid label
                ## positions to a loss computation — at best wasted compute,
                ## at worst a NaN/divide-by-zero depending on how the loss
                ## averages over tokens. Dropping here, at load time, is
                ## cheaper than guarding every downstream consumer.
            rows.append((words, tags))
            # Append this sentence as a (words, tags) tuple to the result list.
    return rows
    # Return the full list of parsed (and filtered) sentence pairs for this split.


def tag_distribution(rows: list[Sentence]) -> Counter:
    """Count tag occurrences across the given rows."""
    # Docstring: documents that this produces a frequency count of BIO tags,
    # useful for inspecting class imbalance in a dataset split.
    counts: Counter = Counter()
    # Initialize an empty Counter (a dict defaulting missing keys to 0) to
    # accumulate tag frequencies.
    for _, tags in rows:
        # Iterate every (words, tags) pair, discarding the words (unused
        # here, hence the `_` placeholder) and keeping the tag list.
        counts.update(tags)
        # Counter.update on an iterable increments the count for each tag
        # string it sees, effectively tallying tag frequency across all rows.
    return counts
    # Return the completed Counter mapping each tag string to how many times
    # it occurred across the whole input.
