"""
Prompt templates cho Text-to-Triples & Entity-Guided CoT Prompting.
Owner: Nguyễn Chí Phong (24020268)
"""

import json
import re
from pathlib import Path

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|im_end\|>|<\|im_start\|>(?:system|user|assistant)?|<\|endoftext\|>"
    r"|<\|eot_id\|>|<\|begin_of_text\|>|<\|start_header_id\|>(?:system|user|assistant)?<\|end_header_id\|>"
    r"|<\|end\|>|<\|assistant\|>|<\|user\|>|<\|system\|>|\[BOS\]|\[EOS\]|<s>|</s>",
    flags=re.IGNORECASE,
)
_PAREN_TRIPLE_RE = re.compile(r"\(([^()]+?\|[^()]+?\|[^()]+?)\)")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def clean_entity(value: str) -> str:
    """Chuẩn hóa tên thực thể/giá trị: bỏ gạch dưới, dấu ngoặc kép thừa và khoảng trắng thừa."""
    text = value.replace("_", " ").strip()
    text = " ".join(text.split())
    while len(text) >= 2 and (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("''") and text.endswith("''"))
        or (text.startswith("'") and text.endswith("'"))
    ):
        if text.startswith("''") and text.endswith("''") and len(text) >= 4:
            text = text[2:-2].strip()
        else:
            text = text[1:-1].strip()
    if text.startswith("''"):
        text = text[2:].strip()
    if text.endswith("''"):
        text = text[:-2].strip()
    if text.startswith('"') or text.endswith('"'):
        text = text.strip('"').strip()
    return " ".join(text.split())


def clean_predicate(value: str) -> str:
    """Chuẩn hóa tên quan hệ (giữ nguyên định dạng camelCase của WebNLG+)."""
    text = value.strip().strip('"').strip("'").strip()
    return " ".join(text.split())


def clean_triple(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    """Chuẩn hóa một bộ ba (Subject, Predicate, Object)."""
    return (clean_entity(subject), clean_predicate(predicate), clean_entity(obj))


def parse_llm_triples(raw_output: str) -> tuple[list[tuple[str, str, str]], bool]:
    """Strip chat template/special tokens của LLM về danh sách (s, p, o) đã deduplicate."""
    if not raw_output or not raw_output.strip():
        return [], True

    text = _THINK_BLOCK_RE.sub("", raw_output)
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = text.replace("<sep>", "\n")

    parsed: list[tuple[str, str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if line.lower().startswith(("entities:", "triples:", "output:", "answer:")):
            continue
        line = _BULLET_PREFIX_RE.sub("", line).strip()

        paren_matches = _PAREN_TRIPLE_RE.findall(line)
        if paren_matches:
            for match in paren_matches:
                parts = [part.strip() for part in match.split("|")]
                if len(parts) == 3:
                    parsed.append(clean_triple(parts[0], parts[1], parts[2]))
            continue

        stripped = line[1:-1].strip() if line.startswith("(") and line.endswith(")") else line
        parts = [part.strip() for part in stripped.split("|")]
        if len(parts) == 3:
            parsed.append(clean_triple(parts[0], parts[1], parts[2]))

    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for s, p, o in parsed:
        if not s or not p or not o:
            continue
        key = (s.lower(), p.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            unique.append((s, p, o))
    return unique, len(unique) == 0


SYSTEM_PROMPT_STANDARD = (
    "You are an expert Information Extraction system. "
    "Extract all factual relational triples from the input text strictly in the format "
    "(Subject | Predicate | Object), one triple per line. "
    "Do not hallucinate or add any extra explanation."
)

SYSTEM_PROMPT_COT = (
    "You are an expert Information Extraction system. "
    "First list the key entities and values present in the input text, "
    "then extract all factual relational triples strictly in the format "
    "(Subject | Predicate | Object), one triple per line. Do not hallucinate."
)

DEFAULT_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "text": "Abilene, Texas is in the United States.",
        "triples": [("Abilene, Texas", "country", "United States")],
        "entities": ["Abilene, Texas", "United States"],
    },
    {
        "text": (
            "546m above sea level, the length of the 1st runway at "
            "Abilene Regional airport is 3678 feet."
        ),
        "triples": [
            ("Abilene Regional Airport", "1stRunwayLengthFeet", "3678"),
            ("Abilene Regional Airport", "elevationAboveTheSeaLevel", "546"),
        ],
        "entities": ["Abilene Regional Airport", "3678", "546"],
    },
    {
        "text": (
            "Abilene Regional Airport serves the city of Abilene in "
            "Jones County, Texas, United States."
        ),
        "triples": [
            ("Abilene Regional Airport", "cityServed", "Abilene, Texas"),
            ("Abilene, Texas", "country", "United States"),
            ("Abilene, Texas", "isPartOf", "Jones County, Texas"),
        ],
        "entities": [
            "Abilene Regional Airport",
            "Abilene, Texas",
            "United States",
            "Jones County, Texas",
        ],
    },
]


def format_prompt(text: str, use_cot: bool = False) -> str:
    cleaned = text.strip()
    if use_cot:
        return (
            f'Text: "{cleaned}"\nIdentify the entities first, then output all relational triples.'
        )
    return f'Text: "{cleaned}"'


def format_target(
    triples: list[tuple[str, str, str]],
    entities: list[str] | None = None,
    use_cot: bool = False,
) -> str:
    triple_lines = "\n".join(f"({s} | {p} | {o})" for s, p, o in triples)
    if not use_cot:
        return triple_lines

    if entities is None:
        seen: set[str] = set()
        entities = []
        for s, _, o in triples:
            for ent in (s, o):
                if ent.lower() not in seen:
                    seen.add(ent.lower())
                    entities.append(ent)

    entity_str = "; ".join(entities)
    return f"Entities: {entity_str}\nTriples:\n{triple_lines}"


def build_messages(
    text: str,
    triples: list[tuple[str, str, str]] | None = None,
    entities: list[str] | None = None,
    use_cot: bool = False,
    few_shot_examples: list[dict] | None = None,
) -> list[dict[str, str]]:
    system_prompt = SYSTEM_PROMPT_COT if use_cot else SYSTEM_PROMPT_STANDARD
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if few_shot_examples:
        for ex in few_shot_examples:
            ex_triples = [tuple(t) for t in ex["triples"]]
            messages.append({"role": "user", "content": format_prompt(ex["text"], use_cot=use_cot)})
            messages.append(
                {
                    "role": "assistant",
                    "content": format_target(
                        ex_triples,
                        entities=ex.get("entities"),
                        use_cot=use_cot,
                    ),
                }
            )

    messages.append({"role": "user", "content": format_prompt(text, use_cot=use_cot)})
    if triples is not None:
        messages.append(
            {
                "role": "assistant",
                "content": format_target(triples, entities=entities, use_cot=use_cot),
            }
        )
    return messages


def render_chat_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    msgs = [dict(m) for m in messages]
    template_str = str(getattr(tokenizer, "chat_template", "") or "")
    if (
        "detailed thinking off" in template_str
        and msgs
        and msgs[0].get("role") == "system"
        and "detailed thinking" not in msgs[0].get("content", "")
    ):
        msgs[0]["content"] = "detailed thinking off\n\n" + msgs[0]["content"]

    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def resolve_split_path(processed_dir: str | Path, split: str) -> Path:
    base = Path(processed_dir)
    candidates_map = {
        "train": ["train.json", "train.jsonl"],
        "val": ["val.json", "validation.json", "validation.jsonl", "val.jsonl"],
        "validation": ["val.json", "validation.json", "validation.jsonl", "val.jsonl"],
        "test": ["test.json", "test.jsonl"],
    }
    candidates = candidates_map.get(split, [f"{split}.json", f"{split}.jsonl"])
    for name in candidates:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu cho split '{split}' trong {base} (đã thử: {candidates})"
    )


def load_webnlg_records(data_path: str | Path, max_samples: int | None = None) -> list[dict]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    if max_samples is not None and len(records) >= max_samples:
                        break
        return records

    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return records[:max_samples] if max_samples is not None else records


def normalize_and_sort_triples(
    text: str,
    raw_triples: list,
    sort_by_subject: bool = True,
) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    cleaned: list[tuple[str, str, str]] = []
    for triple in raw_triples:
        if len(triple) != 3:
            continue
        s, p, o = clean_triple(str(triple[0]), str(triple[1]), str(triple[2]))
        if not s or not p or not o:
            continue
        key = (s.lower(), p.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            cleaned.append((s, p, o))

    if not sort_by_subject:
        return cleaned

    text_lower = text.lower()

    def _sort_key(item: tuple[str, str, str]) -> tuple[int, str, str, str]:
        subj, pred, obj = item
        pos = text_lower.find(subj.lower())
        if pos < 0:
            pos = len(text_lower) + 1
        return (pos, subj.lower(), pred.lower(), obj.lower())

    return sorted(cleaned, key=_sort_key)
