"""
Unit tests cơ bản cho Triplet Parser & Evaluation Metrics.
Owner: Bùi Thị Bích Phượng (24021601)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.parser import parse_triplet, deduplicate_triples, normalize_triple


def test_parse_valid_triplet():
    text = "(Alan Bean | mission | Apollo 12)"
    result = parse_triplet(text)
    assert result == [
        ("Alan Bean", "mission", "Apollo 12")
    ]

def test_parse_multiple_triples():
    text = """
    (Alan Bean | mission | Apollo 12)
    (Apollo 12 | operator | NASA)
    """
    result = parse_triplet(text)
    assert len(result) == 2

def test_parse_special_tokens():
    text = """
    [BOS]
    (Alan Bean | mission | Apollo 12)
    [EOS]
    """
    result = parse_triplet(text)
    assert len(result) == 1

def test_invalid_triplet():
    text = "(Alan Bean | mission)"
    assert parse_triplet(text) == []

def test_empty_input():
    assert parse_triplet("") == []

def test_normalize_triple():
    triple = ("Alan Bean", "mission", "Apollo 12")
    normalized = normalize_triple(triple)
    assert normalized == ("alan bean", "mission", "apollo 12")

def test_deduplicate_triples():
    triples = [
        ("Alan Bean", "mission", "Apollo 12"),
        ("Alan Bean", "mission", "Apollo 12"),
        ("Apollo 12", "operator", "NASA")
    ]
    deduplicated = deduplicate_triples(triples)
    assert len(deduplicated) == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))