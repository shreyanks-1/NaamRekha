"""Shared NaamRekha infrastructure: loader, evaluator, results logger."""
# Module docstring: documents the purpose of this package — it re-exports the
# three building blocks (data loading, evaluation, results logging) used by
# every phase script/notebook in the project, so they share one implementation.

# Blank line separating the module docstring from the imports below (PEP 8 style).

from .data import (
    # Relative import from the sibling module `data.py` inside this package;
    # pulls in everything the rest of the pipeline needs to load the
    # Naamapadam NER dataset and work with its label scheme.
    LABELS,
    # LABELS: the fixed ordered list of BIO tag strings (e.g. "O", "B-PER", ...)
    # used as the single source of truth for label order across the project.
    REPO_ID,
    # REPO_ID: the Hugging Face Hub dataset repo identifier the loader downloads from.
    SPLITS,
    # SPLITS: the tuple of valid dataset split names ("train", "val", "test").
    id2label,
    # id2label: dict mapping integer label ids back to their string tag name.
    label2id,
    # label2id: dict mapping string tag name to its integer id (inverse of id2label).
    load_split,
    # load_split: function that downloads/parses one dataset split into (words, tags) rows.
    tag_distribution,
    # tag_distribution: function that counts how many times each tag occurs in a dataset split.
)
from .eval import evaluate
# Import the `evaluate` function from eval.py, which computes entity-level
# F1 scores (overall and per entity type) using seqeval.
from .results import COLUMNS, RESULTS_PATH, log_result
# Import from results.py: COLUMNS (the fixed CSV column order), RESULTS_PATH
# (the default path to the results CSV file), and log_result (the function
# that appends one experiment run's scores to that CSV).

__all__ = [
    # __all__ explicitly declares this package's public API: these are the
    # only names that `from src import *` will expose, and it documents at a
    # glance everything this shared module offers to notebooks/scripts.
    "COLUMNS",
    # Re-exported: the list of column names for the results CSV.
    "LABELS",
    # Re-exported: the ordered list of BIO NER tags.
    "REPO_ID",
    # Re-exported: the Hugging Face dataset repo id string.
    "RESULTS_PATH",
    # Re-exported: default filesystem path for results.csv.
    "SPLITS",
    # Re-exported: the valid split name tuple.
    "evaluate",
    # Re-exported: the F1 evaluation function.
    "id2label",
    # Re-exported: int -> tag string lookup dict.
    "label2id",
    # Re-exported: tag string -> int lookup dict.
    "load_split",
    # Re-exported: the dataset split loader function.
    "log_result",
    # Re-exported: the CSV-appending results logger function.
    "tag_distribution",
    # Re-exported: the tag-frequency counter function.
]
