from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
THINK_SPECIAL_TOKENS = (THINK_START_TOKEN, THINK_END_TOKEN)


@dataclass(frozen=True)
class MathExample:
    question: str
    answer: str


def normalize_sft_example(
    example: Mapping[str, Any],
) -> tuple[str, list[str], str]:
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    raw_steps = example.get("steps", [])

    if isinstance(raw_steps, str):
        steps = [raw_steps.strip()]
    elif isinstance(raw_steps, Sequence):
        # Preserve empty entries: SIM-CoT tokenizes every listed step as
        # ``step + "\n"``, so an empty step still contributes a newline token.
        steps = [str(step).strip() for step in raw_steps]
    else:
        raise TypeError("'steps' must be a string or a sequence of strings")

    if not question:
        raise ValueError("SFT example has an empty question")
    if not answer:
        raise ValueError("SFT example has an empty answer")

    return question, steps, answer


def format_sft_example(example: Mapping[str, Any]) -> tuple[str, str]:
    """Return the exact explicit-CoT prompt/completion pair used for SFT."""
    question, steps, answer = normalize_sft_example(example)

    # The prompt newline belongs to the masked prefix. Everything from the
    # opening reasoning boundary through the final answer is supervised.
    prompt = question + "\n"
    completion_lines = [
        THINK_START_TOKEN,
        *steps,
        THINK_END_TOKEN,
        answer,
    ]
    completion = "\n".join(completion_lines)
    return prompt, completion


def tokenize_sft_batch(
    batch: Mapping[str, list[Any]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenize a batch and mask the question tokens from the language-model loss."""
    encoded: dict[str, list[list[int]]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id")

    for question, steps, answer in zip(
        batch["question"], batch["steps"], batch["answer"], strict=True
    ):
        question, steps, answer = normalize_sft_example(
            {"question": question, "steps": steps, "answer": answer}
        )
        prompt_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
        completion_ids = tokenizer.encode(
            THINK_START_TOKEN + "\n", add_special_tokens=False
        )
        completion_ids += [
            token_id
            for step in steps
            for token_id in tokenizer.encode(step + "\n", add_special_tokens=False)
        ]
        completion_ids += tokenizer.encode(
            THINK_END_TOKEN + "\n", add_special_tokens=False
        )
        completion_ids += tokenizer.encode(
            answer, add_special_tokens=False
        )
        input_ids = prompt_ids + completion_ids + [eos_token_id]

        # Dropping over-length examples preserves both the full problem and final
        # answer; right truncation would silently remove the supervision target.
        if len(input_ids) > max_length:
            continue

        labels = [-100] * len(prompt_ids) + completion_ids + [eos_token_id]
        encoded["input_ids"].append(input_ids)
        encoded["attention_mask"].append([1] * len(input_ids))
        encoded["labels"].append(labels)

    return encoded


def find_parquet_files(data_root: Path, split: str) -> list[str]:
    paths = sorted((data_root / "gsm8k-aug" / "data").glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No GSM8K-Aug {split!r} parquet files under "
            f"{data_root / 'gsm8k-aug' / 'data'}"
        )
    return [str(path) for path in paths]


def load_gsm8k_aug(
    data_root: Path,
    cache_dir: Path,
    splits: Sequence[str] = ("train", "validation"),
) -> Any:
    from datasets import load_dataset

    data_files = {split: find_parquet_files(data_root, split) for split in splits}
    return load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=str(cache_dir),
    )


def random_partition_indices(
    num_examples: int,
    num_parts: int,
    seed: int,
) -> list[list[int]]:
    """Randomly partition dataset row indices into balanced, reproducible parts."""
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")
    if num_parts < 2:
        raise ValueError("num_parts must be at least 2")
    if num_parts > num_examples:
        raise ValueError("num_parts cannot exceed num_examples")

    shuffled_indices = list(range(num_examples))
    random.Random(seed).shuffle(shuffled_indices)

    base_size, larger_part_count = divmod(num_examples, num_parts)
    partitions: list[list[int]] = []
    start = 0
    for part_index in range(num_parts):
        part_size = base_size + int(part_index < larger_part_count)
        partitions.append(shuffled_indices[start : start + part_size])
        start += part_size
    return partitions


def select_random_dataset_parts(
    dataset: Any,
    num_parts: int,
    selected_parts: Sequence[int],
    seed: int,
) -> tuple[Any, list[int]]:
    """Select one-based random partitions from a Hugging Face Dataset."""
    parts = list(selected_parts)
    if not parts:
        raise ValueError("selected_parts must contain at least one part")
    if len(set(parts)) != len(parts):
        raise ValueError("selected_parts must not contain duplicates")
    invalid_parts = [part for part in parts if part < 1 or part > num_parts]
    if invalid_parts:
        raise ValueError(
            f"selected_parts must be between 1 and {num_parts}; got {invalid_parts}"
        )

    partitions = random_partition_indices(len(dataset), num_parts, seed)
    selected_indices = [
        row_index
        for part in parts
        for row_index in partitions[part - 1]
    ]
    return dataset.select(selected_indices), [len(partition) for partition in partitions]


def _read_json_array(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def load_math_examples(
    dataset_name: str,
    data_root: Path,
    cache_dir: Path,
) -> list[MathExample]:
    """Load one of the local math evaluation sets into a common schema."""
    if dataset_name == "gsm8k":
        dataset = load_gsm8k_aug(data_root, cache_dir, splits=("test",))["test"]
        return [
            MathExample(str(row["question"]).strip(), str(row["answer"]).strip())
            for row in dataset
        ]

    if dataset_name == "gsm-hard":
        rows = _read_jsonl(data_root / "gsm-hard" / "gsmhardv2.jsonl")
        return [
            MathExample(str(row["input"]).strip(), str(row["target"]).strip())
            for row in rows
        ]

    if dataset_name == "multi-arith":
        rows = _read_json_array(data_root / "MultiArith" / "test.json")
        return [
            MathExample(str(row["question"]).strip(), str(row["final_ans"]).strip())
            for row in rows
        ]

    if dataset_name == "svamp":
        # SIM-CoT evaluates SVAMP on the concatenation of its official train and
        # test files (700 + 300 examples).
        rows = _read_json_array(data_root / "SVAMP" / "train.json")
        rows += _read_json_array(data_root / "SVAMP" / "test.json")
        return [
            MathExample(
                " ".join(
                    part
                    for part in (
                        str(row["Body"]).strip(),
                        str(row["Question"]).strip(),
                    )
                    if part
                ),
                str(row["Answer"]).strip(),
            )
            for row in rows
        ]

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_numeric_answer(text: str) -> Decimal | None:
    """Extract the final numeric value from a generated completion."""
    normalized = text.replace("−", "-")
    matches = NUMBER_PATTERN.findall(normalized)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def extract_answer_after_think(text: str) -> str | None:
    """Return only the answer segment following the first closing think token."""
    _, separator, answer_text = text.partition(THINK_END_TOKEN)
    if not separator:
        return None
    return answer_text.strip()


def numeric_answers_equal(prediction: Decimal | None, gold: str) -> bool:
    if prediction is None:
        return False
    gold_value = extract_numeric_answer(str(gold))
    return gold_value is not None and prediction == gold_value


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
