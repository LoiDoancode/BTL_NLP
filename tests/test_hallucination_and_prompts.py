"""
Unit tests cho Entity Hallucination Filter (Phượng) và Prompt/Parser LLMs < 6B (Phong).
"""

from eval.hallucination_filter import entity_exists, filter_hallucinated_triples
from models.llm_under6b.prompts import (
    build_messages,
    clean_entity,
    format_target,
    normalize_and_sort_triples,
    parse_llm_triples,
)


def test_hallucination_filter_handles_underscores_and_rejects_fabricated() -> None:
    source = "Aarhus Airport serves the city of Aarhus, Denmark."
    assert entity_exists("Aarhus_Airport", source)
    assert entity_exists("Aarhus Airport", source)
    assert not entity_exists("Copenhagen", source)

    triples = [
        ("Aarhus_Airport", "cityServed", "Aarhus, Denmark"),
        ("Aarhus Airport", "country", "Sweden"),
    ]
    res = filter_hallucinated_triples(triples, source)
    assert res["valid"] == [("Aarhus_Airport", "cityServed", "Aarhus, Denmark")]
    assert len(res["rejected"]) == 1
    assert res["rejected"][0]["object_found"] is False


def test_llm_prompts_clean_parse_and_sort() -> None:
    assert clean_entity('  "Aarhus_Airport"  ') == "Aarhus Airport"

    raw = (
        "<think>reasoning</think>\n"
        "Entities: Aarhus Airport; Aarhus, Denmark\n"
        "Triples:\n"
        "- (Aarhus_Airport | cityServed | Aarhus, Denmark)\n"
        "(Aarhus Airport | cityServed | Aarhus, Denmark)\n"
    )
    triples, is_unparseable = parse_llm_triples(raw)
    assert not is_unparseable
    assert triples == [("Aarhus Airport", "cityServed", "Aarhus, Denmark")]

    cot_target = format_target(triples, use_cot=True)
    assert "Entities:" in cot_target and "Triples:" in cot_target

    msgs = build_messages("Aarhus Airport serves Aarhus.", triples=triples, use_cot=False)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "assistant"

    sorted_triples = normalize_and_sort_triples(
        "Alpha founded Beta in Gamma.",
        [("Beta", "location", "Gamma"), ("Alpha", "founderOf", "Beta")],
        sort_by_subject=True,
    )
    assert sorted_triples[0][0] == "Alpha"
