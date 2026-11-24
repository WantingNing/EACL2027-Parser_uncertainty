#!/usr/bin/env python3
"""Run IsaNLP-RST on source documents and export normalized per-document JSON."""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from razdel import sentenize, tokenize

try:
    from preprocess.isanlp_trace_parser import TraceableParser, build_relation_label_set
except ModuleNotFoundError:
    from isanlp_trace_parser import TraceableParser, build_relation_label_set


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "parsers" / "rst"
GCDC_PREFIXES = ("clinton", "enron", "yahoo", "yelp")
PARSER_TYPES = ("rstdt", "gumrrg", "rstreebank")
DEFAULT_MODEL_NAME = "tchewik/isanlp_rst_v3"


@dataclass(frozen=True)
class DocumentRecord:
    source_name: str
    group_name: str
    doc_id: str
    text: str
    raw_record: dict[str, Any]


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class CharacterSpan:
    start: int
    end: int


class DocumentSource(ABC):
    """Abstract source for parser input documents."""

    @abstractmethod
    def iter_documents(self) -> Iterable[DocumentRecord]:
        raise NotImplementedError


class GCDCDocumentSource(DocumentSource):
    def __init__(self, prefixes: list[str], input_root: Path):
        self.prefixes = prefixes
        self.input_root = input_root

    def iter_documents(self) -> Iterable[DocumentRecord]:
        for prefix in self.prefixes:
            input_path = resolve_gcdc_input_path(self.input_root, prefix)
            with input_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    record = json.loads(line)
                    text_id = str(record.get("text_id", record.get("id", line_number)))
                    doc_id = f"{prefix}_{text_id}"
                    yield DocumentRecord(
                        source_name="gcdc",
                        group_name=prefix,
                        doc_id=doc_id,
                        text=str(record["text"]),
                        raw_record=record,
                    )


class JsonlDocumentSource(DocumentSource):
    def __init__(
        self,
        input_path: Path,
        source_name: str,
        group_name: str,
        text_field: str,
        doc_id_field: str,
    ):
        self.input_path = input_path
        self.source_name = source_name
        self.group_name = group_name
        self.text_field = text_field
        self.doc_id_field = doc_id_field

    def iter_documents(self) -> Iterable[DocumentRecord]:
        with self.input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                doc_id = str(record.get(self.doc_id_field, line_number))
                yield DocumentRecord(
                    source_name=self.source_name,
                    group_name=self.group_name,
                    doc_id=doc_id,
                    text=str(record[self.text_field]),
                    raw_record=record,
                )


class ParserAdapter(ABC):
    @abstractmethod
    def parse(self, text: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parser_version(self) -> str:
        raise NotImplementedError

    def relation_label_set(self) -> list[str] | None:
        return None


class LocalIsaNLPParserAdapter(ParserAdapter):
    def __init__(self, parser_type: str, model_name: str, cuda_device: int):
        self._parser = TraceableParser(
            hf_model_name=model_name,
            hf_model_version=parser_type,
            cuda_device=cuda_device,
        )

    def parse(self, text: str) -> dict[str, Any]:
        return self._parser(text)

    def parser_version(self) -> str:
        try:
            version = metadata.version("isanlp-rst")
        except metadata.PackageNotFoundError:
            version = "unknown"
        return f"isanlp-rst@{version}"

    def relation_label_set(self) -> list[str] | None:
        return infer_relation_inventory(self._parser)


class OffsetNormalizer:
    """Convert parser character offsets to Python's zero-based exclusive spans."""

    def __init__(self, text: str, root_node: Any):
        self.text_length = len(text)
        self.mode = self._detect_mode(root_node)

    def _detect_mode(self, node: Any) -> str:
        start = require_int(get_node_attr(node, "start"), "root.start")
        end = require_int(get_node_attr(node, "end"), "root.end")

        if start == 1 and end == self.text_length:
            return "one_based_inclusive"
        if start == 0 and end == self.text_length:
            return "zero_based_exclusive"
        if start == 0 and end == self.text_length - 1:
            return "zero_based_inclusive"

        for mode in ("one_based_inclusive", "zero_based_exclusive", "zero_based_inclusive"):
            span = self._convert(start, end, mode)
            if span.start == 0 and span.end == self.text_length:
                return mode

        # Some parsers return a root span for the parsed text region rather than
        # the full original string. This shows up in CoheSentia examples with
        # trailing invisible/control characters or parser-trimmed suffixes. In
        # that case the offset convention is still inferable from the start
        # index, and the later EDU-to-token range assignment deliberately
        # expands the final EDU to cover all document tokens.
        if start == 0 and 0 <= end <= self.text_length:
            return "zero_based_exclusive"
        if start == 0 and 0 <= end + 1 <= self.text_length:
            return "zero_based_inclusive"
        if start == 1 and 0 <= end <= self.text_length:
            return "one_based_inclusive"

        raise ValueError(
            f"Unable to infer parser offset convention from root span ({start}, {end}) "
            f"for text length {self.text_length}."
        )

    def normalize(self, node: Any) -> CharacterSpan:
        start = require_int(get_node_attr(node, "start"), "node.start")
        end = require_int(get_node_attr(node, "end"), "node.end")
        return self._convert(start, end, self.mode)

    def _convert(self, start: int, end: int, mode: str) -> CharacterSpan:
        if mode == "one_based_inclusive":
            span = CharacterSpan(start=max(start - 1, 0), end=end)
        elif mode == "zero_based_exclusive":
            span = CharacterSpan(start=start, end=end)
        elif mode == "zero_based_inclusive":
            span = CharacterSpan(start=start, end=end + 1)
        else:  # pragma: no cover - guarded by _detect_mode
            raise ValueError(f"Unknown offset mode: {mode}")

        if not (0 <= span.start <= span.end <= self.text_length):
            raise ValueError(
                f"Invalid normalized span {span} produced from ({start}, {end}) with mode {mode}."
            )
        return span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse source documents with IsaNLP-RST and export one normalized JSON file "
            "per document."
        )
    )
    parser.add_argument("--source", choices=("gcdc", "jsonl"), default="gcdc")
    parser.add_argument("--parser-type", choices=PARSER_TYPES, default="rstdt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--gcdc-root", type=Path, default=REPO_ROOT / "data" / "GCDC")
    parser.add_argument(
        "--gcdc-prefix",
        nargs="+",
        choices=GCDC_PREFIXES,
        default=list(GCDC_PREFIXES),
        help="One or more GCDC domains to parse.",
    )

    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--source-name", default="custom")
    parser.add_argument("--group-name", default="default")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--doc-id-field", default="text_id")

    parser.add_argument("--hf-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument("--parser-version")
    parser.add_argument("--relation-label-set-file", type=Path)

    parser.add_argument(
        "--tokenizer",
        choices=("razdel", "regex"),
        default="razdel",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_source(args: argparse.Namespace) -> DocumentSource:
    if args.source == "gcdc":
        return GCDCDocumentSource(
            prefixes=args.gcdc_prefix,
            input_root=args.gcdc_root,
        )

    if args.input_jsonl is None:
        raise ValueError("`--input-jsonl` is required when `--source jsonl` is used.")

    return JsonlDocumentSource(
        input_path=args.input_jsonl,
        source_name=args.source_name,
        group_name=args.group_name,
        text_field=args.text_field,
        doc_id_field=args.doc_id_field,
    )


def build_parser(args: argparse.Namespace) -> ParserAdapter:
    return LocalIsaNLPParserAdapter(
        parser_type=args.parser_type,
        model_name=args.hf_model_name,
        cuda_device=args.cuda_device,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    source = build_source(args)
    parser = build_parser(args)

    relation_label_set = load_relation_label_set(args.relation_label_set_file)
    if relation_label_set is None:
        relation_label_set = parser.relation_label_set()

    tokenizer_name = args.tokenizer
    parser_version = args.parser_version or parser.parser_version()
    output_dir = args.output_root / f"isanlp-{args.parser_type}"
    # Buffer payloads only when relation_label_set is unknown and needs backfilling
    # after all documents are parsed.  When relation_label_set is already known,
    # write each document to disk immediately to avoid holding the full dataset in RAM.
    buffer_for_backfill = relation_label_set is None
    pending_writes: list[tuple[Path, dict[str, Any]]] = []
    observed_relations: set[str] = set()
    document_count = 0

    for document in source.iter_documents():
        output_path = build_output_path(output_dir, document)
        if output_path.exists() and not args.overwrite:
            logging.info("Skipping existing file: %s", output_path)
            continue

        logging.info("Parsing %s", document.doc_id)
        result = parser.parse(document.text)
        payload = build_output_payload(
            document=document,
            parser_result=result,
            tokenizer_name=tokenizer_name,
            parser_version=parser_version,
            relation_label_set=relation_label_set or [],
            tokenizer_mode=args.tokenizer,
        )

        if buffer_for_backfill:
            observed_relations.update(node["relation_label"] for node in payload["tree"]["nodes"])
            pending_writes.append((output_path, payload))
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        document_count += 1

    if buffer_for_backfill:
        relation_label_set = sorted(observed_relations)
        for _, payload in pending_writes:
            payload["parser_metadata"]["relation_label_set"] = relation_label_set
        for output_path, payload in pending_writes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

    metadata_path = output_dir / "parser_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "parser_type": args.parser_type,
                "parser_version": parser_version,
                "tokenizer_name": tokenizer_name,
                "relation_label_set": relation_label_set or [],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("Wrote %s documents to %s", document_count, output_dir)


def build_output_payload(
    document: DocumentRecord,
    parser_result: dict[str, Any],
    tokenizer_name: str,
    parser_version: str,
    relation_label_set: list[str],
    tokenizer_mode: str,
) -> dict[str, Any]:
    tokens = tokenize_with_offsets(document.text, tokenizer_mode)
    sentences = get_sentence_spans(document.text, tokens, tokenizer_mode)
    paragraphs = get_paragraph_spans(document.text, tokens)
    tree = build_tree(document.text, parser_result, tokens)

    return {
        "doc_id": document.doc_id,
        "tokens": [token.text for token in tokens],
        "sentences": sentences,
        "paragraphs": paragraphs,
        "edus": tree["edus"],
        "tree": {
            "root_node_id": tree["root_node_id"],
            "nodes": tree["nodes"],
        },
        "parser_metadata": {
            "relation_label_set": relation_label_set,
            "parser_version": parser_version,
            "tokenizer_name": tokenizer_name,
        },
    }


def build_tree(
    text: str,
    parser_result: dict[str, Any],
    tokens: list[TokenSpan],
) -> dict[str, Any]:
    rst_value = parser_result.get("rst", [])
    if isinstance(rst_value, list):
        root = rst_value[0] if rst_value else None
    else:
        root = rst_value

    if root is None:
        edus = fallback_edus(tokens)
        return {"edus": edus, "nodes": [], "root_node_id": None}

    normalizer = OffsetNormalizer(text, root)
    leaf_nodes: list[Any] = []
    collect_leaves(root, leaf_nodes)

    if not leaf_nodes:
        edus = fallback_edus(tokens)
        return {"edus": edus, "nodes": [], "root_node_id": None}

    leaf_nodes.sort(key=lambda node: (normalizer.normalize(node).start, normalizer.normalize(node).end))
    leaf_id_map = {id(node): edu_id for edu_id, node in enumerate(leaf_nodes)}
    leaf_spans = [normalizer.normalize(leaf) for leaf in leaf_nodes]
    edus: list[dict[str, int]] = []

    token_ranges = leaf_spans_to_token_ranges(leaf_spans, tokens)
    for edu_id, (token_start, token_end) in enumerate(token_ranges):
        edus.append(
            {
                "edu_id": edu_id,
                "token_start": token_start,
                "token_end": token_end,
            }
        )

    validate_edu_coverage(edus, len(tokens))

    internal_nodes: list[dict[str, Any]] = []
    assigned_node_ids: dict[int, int] = {}
    used_node_ids: set[int] = set()
    synthetic_node_id = max([edu["edu_id"] for edu in edus], default=-1) + 100

    def next_node_id(node: Any) -> int:
        nonlocal synthetic_node_id
        object_id = id(node)
        if object_id in assigned_node_ids:
            return assigned_node_ids[object_id]

        candidate = get_node_attr(node, "id")
        if isinstance(candidate, int) and candidate not in used_node_ids:
            node_id = candidate
        else:
            while synthetic_node_id in used_node_ids:
                synthetic_node_id += 1
            node_id = synthetic_node_id
            synthetic_node_id += 1

        assigned_node_ids[object_id] = node_id
        used_node_ids.add(node_id)
        return node_id

    def build_node_ref(node: Any) -> tuple[dict[str, Any], tuple[int, int]]:
        left = get_node_attr(node, "left")
        right = get_node_attr(node, "right")

        if left is None and right is None:
            edu_id = leaf_id_map[id(node)]
            return {"kind": "leaf", "id": edu_id}, (edu_id, edu_id)

        if left is None or right is None:
            raise ValueError("Internal parser node must have both left and right children.")

        left_ref, left_span = build_node_ref(left)
        right_ref, right_span = build_node_ref(right)
        node_id = next_node_id(node)
        edu_span = (left_span[0], right_span[1])
        node_payload = {
            "node_id": node_id,
            "is_leaf": False,
            "left_child": left_ref,
            "right_child": right_ref,
            "span": {
                "edu_start": edu_span[0],
                "edu_end": edu_span[1],
            },
            "relation_label": str(get_node_attr(node, "relation", default="")),
            "nuclearity_label": str(get_node_attr(node, "nuclearity", default="")),
            "split_entropy": float(get_node_attr(node, "entropy", default=0.0) or 0.0),
            "num_split_candidates": max(edu_span[1] - edu_span[0], 1),
        }

        add_optional_node_fields(node_payload, node)
        internal_nodes.append(node_payload)
        return {"kind": "internal", "id": node_id}, edu_span

    root_ref, _ = build_node_ref(root)
    internal_nodes.sort(key=lambda item: item["node_id"])
    return {
        "edus": edus,
        "nodes": internal_nodes,
        "root_node_id": root_ref["id"] if root_ref["kind"] == "internal" else None,
    }


def add_optional_node_fields(node_payload: dict[str, Any], node: Any) -> None:
    relation_distribution = normalize_distribution(
        get_first_present_attr(
            node,
            (
                "relation_distribution",
                "relation_dist",
                "relation_probs",
                "relation_probabilities",
            ),
        )
    )
    if relation_distribution is not None:
        node_payload["relation_distribution"] = relation_distribution
        relation_entropy = compute_distribution_entropy(relation_distribution)
        if relation_entropy is not None:
            node_payload["relation_entropy"] = relation_entropy

    nuclearity_distribution = normalize_distribution(
        get_first_present_attr(
            node,
            (
                "nuclearity_distribution",
                "nuclearity_dist",
                "nuclearity_probs",
                "nuclearity_probabilities",
            ),
        )
    )
    if nuclearity_distribution is not None:
        node_payload["nuclearity_distribution"] = nuclearity_distribution
        nuclearity_entropy = compute_distribution_entropy(nuclearity_distribution)
        if nuclearity_entropy is not None:
            node_payload["nuclearity_entropy"] = nuclearity_entropy
            node_payload["nuclerity_entropy"] = nuclearity_entropy

    split_distribution = normalize_distribution(
        get_first_present_attr(
            node,
            (
                "split_distribution",
                "split_dist",
                "split_probs",
                "split_probabilities",
            ),
        )
    )
    if split_distribution is not None:
        node_payload["split_distribution"] = split_distribution

    is_artifact = get_first_present_attr(
        node,
        (
            "is_binarization_artifact",
            "binarization_artifact",
            "is_artificial",
        ),
    )
    if is_artifact is not None:
        node_payload["is_binarization_artifact"] = bool(is_artifact)


def build_output_path(output_dir: Path, document: DocumentRecord) -> Path:
    source_dir = sanitize_path_component(document.source_name)
    group_dir = sanitize_path_component(document.group_name)
    filename = sanitize_path_component(document.doc_id) + ".json"
    return output_dir / source_dir / group_dir / filename


def tokenize_with_offsets(text: str, tokenizer_mode: str) -> list[TokenSpan]:
    if not text:
        return []

    if tokenizer_mode == "razdel" and tokenize is not None:
        return [TokenSpan(text=token.text, start=token.start, end=token.stop) for token in tokenize(text)]

    return [
        TokenSpan(text=match.group(0), start=match.start(), end=match.end())
        for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    ]


def get_sentence_spans(
    text: str,
    tokens: list[TokenSpan],
    tokenizer_mode: str,
) -> list[dict[str, int]]:
    if not tokens:
        return []

    if tokenizer_mode == "razdel" and sentenize is not None:
        spans = [CharacterSpan(segment.start, segment.stop) for segment in sentenize(text)]
    else:
        spans = regex_sentence_spans(text)

    return char_spans_to_token_spans(spans, tokens)


def get_paragraph_spans(text: str, tokens: list[TokenSpan]) -> list[dict[str, int]]:
    if not tokens:
        return []

    spans: list[CharacterSpan] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.start()
        if text[start:end].strip():
            spans.append(CharacterSpan(start=start, end=end))
        start = match.end()
    if text[start:].strip():
        spans.append(CharacterSpan(start=start, end=len(text)))

    return char_spans_to_token_spans(spans or [CharacterSpan(0, len(text))], tokens)


def regex_sentence_spans(text: str) -> list[CharacterSpan]:
    spans: list[CharacterSpan] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = match.end()
        if text[start:end].strip():
            spans.append(CharacterSpan(start=start, end=end))
        start = end
    if text[start:].strip():
        spans.append(CharacterSpan(start=start, end=len(text)))
    return spans or [CharacterSpan(0, len(text))]


def char_spans_to_token_spans(
    spans: list[CharacterSpan],
    tokens: list[TokenSpan],
) -> list[dict[str, int]]:
    token_spans: list[dict[str, int]] = []
    for span in spans:
        try:
            token_start, token_end = char_span_to_token_range(span, tokens)
        except ValueError:
            continue
        token_spans.append({"token_start": token_start, "token_end": token_end})
    return token_spans


def char_span_to_token_range(span: CharacterSpan, tokens: list[TokenSpan]) -> tuple[int, int]:
    overlapping = [
        index
        for index, token in enumerate(tokens)
        if token.start < span.end and token.end > span.start
    ]
    if not overlapping:
        raise ValueError(f"Span {span} does not overlap any tokens.")
    return overlapping[0], overlapping[-1]


def leaf_spans_to_token_ranges(
    leaf_spans: list[CharacterSpan],
    tokens: list[TokenSpan],
) -> list[tuple[int, int]]:
    if not leaf_spans:
        return []
    if not tokens:
        raise ValueError("Cannot map non-empty EDU spans onto an empty token list.")

    edu_count = len(leaf_spans)
    token_count = len(tokens)
    if token_count < edu_count:
        raise ValueError(
            f"Token count ({token_count}) is smaller than EDU count ({edu_count}); "
            "cannot assign at least one token to each EDU."
        )

    token_centers = [(token.start + token.end) / 2 for token in tokens]
    boundary_points = [
        (leaf_spans[index].end + leaf_spans[index + 1].start) / 2
        for index in range(edu_count - 1)
    ]

    ranges: list[tuple[int, int]] = []
    start_token = 0
    for boundary_index, boundary_point in enumerate(boundary_points):
        preferred_end = bisect.bisect_right(token_centers, boundary_point) - 1
        min_end = start_token
        remaining_edus = edu_count - boundary_index - 1
        max_end = token_count - remaining_edus - 1
        end_token = min(max(preferred_end, min_end), max_end)
        ranges.append((start_token, end_token))
        start_token = end_token + 1

    ranges.append((start_token, token_count - 1))
    return ranges


def collect_leaves(node: Any, leaves: list[Any]) -> None:
    left = get_node_attr(node, "left")
    right = get_node_attr(node, "right")
    if left is None and right is None:
        leaves.append(node)
        return
    if left is None or right is None:
        raise ValueError("Malformed parser tree: a node has exactly one child.")
    collect_leaves(left, leaves)
    collect_leaves(right, leaves)


def fallback_edus(tokens: list[TokenSpan]) -> list[dict[str, int]]:
    if not tokens:
        return []
    return [{"edu_id": 0, "token_start": 0, "token_end": len(tokens) - 1}]


def validate_edu_coverage(edus: list[dict[str, int]], token_count: int) -> None:
    if not edus:
        if token_count == 0:
            return
        raise ValueError("No EDUs were produced for a non-empty token sequence.")

    if edus[0]["token_start"] != 0:
        raise ValueError("EDUs do not start at token 0.")
    if edus[-1]["token_end"] != token_count - 1:
        raise ValueError("EDUs do not cover the final token.")

    expected_start = 0
    for edu in edus:
        if edu["token_start"] != expected_start:
            raise ValueError("EDUs contain a gap or overlap in token coverage.")
        if edu["token_end"] < edu["token_start"]:
            raise ValueError("EDU token span is invalid.")
        expected_start = edu["token_end"] + 1


def get_node_attr(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def get_first_present_attr(node: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = get_node_attr(node, name, default=None)
        if value is not None:
            return value
    return None


def normalize_distribution(value: Any) -> list[float] | None:
    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None

    distribution = [float(item) for item in value]
    if not distribution:
        return None
    return distribution


def compute_distribution_entropy(distribution: list[float] | None) -> float | None:
    if distribution is None:
        return None

    positive = [float(value) for value in distribution if float(value) > 0.0]
    if not positive:
        return 0.0

    total = sum(positive)
    if total <= 0.0:
        return 0.0

    normalized = [value / total for value in positive]
    return -sum(value * math.log(value) for value in normalized)


def infer_relation_inventory(parser: Any) -> list[str] | None:
    candidates = (
        parser,
        getattr(parser, "predictor", None),
        getattr(getattr(parser, "predictor", None), "model", None),
    )

    for candidate in candidates:
        if candidate is None:
            continue
        for attr_name in (
            "relation_label_set",
            "relation_labels",
            "relation_list",
            "relations",
            "relation_vocab",
            "relation_table",
        ):
            value = getattr(candidate, attr_name, None)
            normalized = normalize_relation_inventory(value)
            if normalized:
                return normalized
    return None


def normalize_relation_inventory(value: Any) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, dict):
        if all(isinstance(key, int) for key in value):
            return [str(value[index]) for index in sorted(value)]
        return sorted(str(key) for key in value)

    if isinstance(value, (list, tuple, set)):
        labels = [str(item) for item in value]
        if labels and any(label.rsplit("_", 1)[-1] in {"NS", "SN", "NN"} for label in labels):
            return build_relation_label_set(labels)
        return labels

    return None


def load_relation_label_set(path: Path | None) -> list[str] | None:
    if path is None:
        return None

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("Relation label set JSON must contain a list of strings.")
        return [str(item) for item in data]

    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_gcdc_input_path(input_root: Path, prefix: str) -> Path:
    direct_path = input_root / f"gcdc_{prefix}" / f"gcdc_{prefix}_all.jsonl"
    if direct_path.exists():
        return direct_path

    fallback_candidates = sorted((input_root / f"gcdc_{prefix}").glob(f"gcdc_{prefix}*_all.jsonl"))
    if fallback_candidates:
        logging.warning(
            "Using fallback GCDC input for prefix `%s`: %s",
            prefix,
            fallback_candidates[0].name,
        )
        return fallback_candidates[0]

    raise FileNotFoundError(
        f"Could not find GCDC input for prefix `{prefix}` under {input_root}."
    )


def sanitize_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def require_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Expected integer value for {field_name}, got {value!r}.")
    return value


if __name__ == "__main__":
    main()
