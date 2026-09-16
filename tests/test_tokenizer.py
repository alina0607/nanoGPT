"""Specification for the character-level tokenizer.

These tests are the acceptance criteria for `tokenizer.py`. Write the
implementation until every test passes:

    python3 -m pytest tests/test_tokenizer.py -v

Required interface
------------------
    class CharTokenizer:
        def __init__(self, text: str) -> None
            Build the vocabulary from `text`.

        vocab_size: int
            Number of distinct characters in the vocabulary.

        def encode(self, s: str) -> list[int]
            Map a string to a list of token ids.

        def decode(self, ids: list[int]) -> str
            Map a list of token ids back to a string.

Design constraints
------------------
1. The vocabulary is every distinct character in `text`, **sorted**. Sorting is
   what makes the ids reproducible: an unsorted `set` gives different ids on
   different runs, so a model trained today would read garbage tomorrow.
2. Ids are consecutive integers starting at 0.
3. `decode(encode(s)) == s` for any string built from the vocabulary.
4. The vocabulary comes from the text passed in, not from a hard-coded table —
   the same class must work when the corpus is swapped.
"""

import pytest

from tokenizer import CharTokenizer

CORPUS_PATH = "data/tinyshakespeare/input.txt"


@pytest.fixture(scope="module")
def text() -> str:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def tok(text: str) -> CharTokenizer:
    return CharTokenizer(text)


def test_vocab_size(tok):
    """tiny Shakespeare contains exactly 65 distinct characters."""
    assert tok.vocab_size == 65


def test_lowest_ids_are_newline_and_space(tok):
    """Sorting by codepoint puts newline (10) before space (32) before the rest."""
    assert tok.encode("\n") == [0]
    assert tok.encode(" ") == [1]


def test_encode_known_string(tok):
    """The worked example from the lecture note."""
    assert tok.encode("First Cit") == [18, 47, 56, 57, 58, 1, 15, 47, 58]


def test_decode_known_ids(tok):
    assert tok.decode([18, 47, 56, 57, 58, 1, 15, 47, 58]) == "First Cit"


@pytest.mark.parametrize(
    "sample",
    [
        "",
        "a",
        "\n",
        "To be, or not to be",
        "All:\nSpeak, speak.\n",
    ],
)
def test_round_trip(tok, sample):
    """decode must invert encode, including on the empty string."""
    assert tok.decode(tok.encode(sample)) == sample


def test_round_trip_on_whole_corpus(tok, text):
    """Nothing in the corpus may be lost or altered."""
    assert tok.decode(tok.encode(text)) == text


def test_encode_is_one_id_per_character(tok, text):
    """Character-level means the sequence length equals the character count."""
    assert len(tok.encode(text)) == 1_115_394


def test_ids_are_in_range(tok, text):
    """Every id must be a valid index into the vocabulary."""
    ids = tok.encode(text[:50_000])
    assert min(ids) >= 0
    assert max(ids) < tok.vocab_size


def test_ids_are_contiguous_from_zero(tok, text):
    """All 65 ids must actually be used — no gaps in the id space."""
    assert set(tok.encode(text)) == set(range(tok.vocab_size))


def test_vocabulary_is_built_from_the_given_text():
    """The class must work on any corpus, not just tiny Shakespeare.

    This is what lets the dataset be swapped later without touching the model.
    """
    other = CharTokenizer("banana")
    assert other.vocab_size == 3  # 'a', 'b', 'n'
    assert other.encode("banana") == [1, 0, 2, 0, 2, 0]
    assert other.decode([1, 0, 2, 0, 2, 0]) == "banana"


def test_two_tokenizers_on_same_text_agree(text):
    """Sorted vocabularies make ids deterministic across instances and runs."""
    a = CharTokenizer(text)
    b = CharTokenizer(text)
    assert a.encode("First Citizen") == b.encode("First Citizen")
