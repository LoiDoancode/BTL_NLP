"""
Unit tests cho Triplet Parser (bao gồm ngoặc lồng nhau, special tokens LLM, chuẩn hóa '_').
Owner: Bùi Thị Bích Phượng (24021601)
"""

from eval.parser import deduplicate_triples, normalize_triple, parse_triplet


def test_parse_valid_triplet() -> None:
    text = "(Alan Bean | mission | Apollo 12)"
    assert parse_triplet(text) == [("Alan Bean", "mission", "Apollo 12")]


def test_parse_multiple_triples() -> None:
    text = """
    (Alan Bean | mission | Apollo 12)
    (Apollo 12 | operator | NASA)
    """
    assert len(parse_triplet(text)) == 2


def test_parse_nested_parentheses_in_entity() -> None:
    text = "(John Smith (politician) | birthPlace | London) | (Aarhus_(city) | country | Denmark)"
    result = parse_triplet(text)
    assert result == [
        ("John Smith (politician)", "birthPlace", "London"),
        ("Aarhus_(city)", "country", "Denmark"),
    ]


def test_parse_special_tokens_and_think_block() -> None:
    text = """
    <think>Find entities first.</think>
    <|im_start|>assistant
    [BOS] (Alan Bean | mission | Apollo 12) [EOS]
    <|eot_id|>
    """
    assert parse_triplet(text) == [("Alan Bean", "mission", "Apollo 12")]


def test_invalid_triplet() -> None:
    assert parse_triplet("(Alan Bean | mission)") == []


def test_empty_input() -> None:
    assert parse_triplet("") == []


def test_normalize_triple_underscores() -> None:
    triple = ("Aarhus_Airport", "cityServed", "Aarhus,_Denmark")
    assert normalize_triple(triple) == ("aarhus airport", "cityserved", "aarhus, denmark")


def test_deduplicate_triples() -> None:
    triples = [
        ("Alan_Bean", "mission", "Apollo_12"),
        ("Alan Bean", "mission", "Apollo 12"),
        ("Apollo 12", "operator", "NASA"),
    ]
    assert len(deduplicate_triples(triples)) == 2
