from __future__ import annotations

import json
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm.auto import trange
from transformers import AutoModel

from graph_cache import get_fold_paths


@dataclass
class DocumentBatch:
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    window_input_ids: torch.Tensor
    window_attention_mask: torch.Tensor
    window_to_global: torch.Tensor
    doc_repr: torch.Tensor
    global_states: torch.Tensor
    global_state_mask: torch.Tensor
    rst_tensors: list[dict[str, Any] | None]
    edu_to_subwords: torch.Tensor
    edu_mask: torch.Tensor
    edu_valid_mask: torch.Tensor
    labels: torch.Tensor
    q_emp: torch.Tensor
    mu_ratings: torch.Tensor
    au_coh: torch.Tensor
    raw_score_5: torch.Tensor
    graphs: list[dict[str, Any]]
    doc_ids: list[str]
    text_ids: list[str]

    def to(self, device: torch.device) -> "DocumentBatch":
        return DocumentBatch(
            doc_input_ids=self.doc_input_ids.to(device),
            doc_attention_mask=self.doc_attention_mask.to(device),
            window_input_ids=self.window_input_ids.to(device),
            window_attention_mask=self.window_attention_mask.to(device),
            window_to_global=self.window_to_global.to(device),
            doc_repr=self.doc_repr.to(device),
            global_states=self.global_states.to(device),
            global_state_mask=self.global_state_mask.to(device),
            rst_tensors=[
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in item.items()
                }
                if item is not None
                else None
                for item in self.rst_tensors
            ],
            edu_to_subwords=self.edu_to_subwords.to(device),
            edu_mask=self.edu_mask.to(device),
            edu_valid_mask=self.edu_valid_mask.to(device),
            labels=self.labels.to(device),
            q_emp=self.q_emp.to(device),
            mu_ratings=self.mu_ratings.to(device),
            au_coh=self.au_coh.to(device),
            raw_score_5=self.raw_score_5.to(device),
            graphs=self.graphs,
            doc_ids=self.doc_ids,
            text_ids=self.text_ids,
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sanitize_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def text_alignment_units(text: str) -> list[tuple[str, int, int]]:
    """Split text into word units and single-punctuation units with char spans."""
    units: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        units.append((match.group(0), match.start(), match.end()))
    return units


def align_tokens_to_text(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    units = text_alignment_units(text)
    spans: list[tuple[int, int]] = []
    unit_cursor = 0
    for token in tokens:
        target = re.sub(r"\s+", "", token.casefold())
        if not target:
            tail = units[unit_cursor][1] if unit_cursor < len(units) else len(text)
            spans.append((tail, tail))
            continue

        matched: tuple[int, int, int] | None = None
        for start_idx in range(unit_cursor, len(units)):
            joined = ""
            end_idx = start_idx
            max_end = min(len(units), start_idx + max(len(target) + 4, 8))
            while end_idx < max_end and len(joined) < len(target):
                joined += units[end_idx][0].casefold()
                end_idx += 1
                if joined == target:
                    matched = (units[start_idx][1], units[end_idx - 1][2], end_idx)
                    break
            if matched is not None:
                break

        if matched is None:
            # The SOTA text-only data builds document text from `sents`.  Some
            # GCDC `sents` omit a trailing fragment that is still present in the
            # raw text and parser tokens.  We keep the SOTA representation
            # unchanged and collapse parser-only tokens to the nearest visible
            # boundary so downstream EDU spans map to the closest available
            # subword rather than failing.
            tail = units[unit_cursor][1] if unit_cursor < len(units) else len(text)
            spans.append((tail, tail))
            continue

        start, end, unit_cursor = matched
        spans.append((start, end))
    return spans


def build_windows(
    tokenizer,
    token_ids: list[int],
    max_length: int,
    window_stride: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must expose bos/eos or cls/sep token ids.")

    content_window = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_window <= 0:
        raise ValueError(f"Unsupported max_length={max_length}.")

    windows_ids: list[list[int]] = []
    windows_mask: list[list[int]] = []
    windows_to_global: list[list[int]] = []
    start = 0
    while True:
        end = min(start + content_window, len(token_ids))
        content_ids = token_ids[start:end]
        prepared = [bos_id] + content_ids + [eos_id]
        attention_mask = [1] * len(prepared)
        mapping = [-1] + list(range(start, end)) + [-1]

        pad_size = max_length - len(prepared)
        prepared = prepared + [tokenizer.pad_token_id] * pad_size
        attention_mask = attention_mask + [0] * pad_size
        mapping = mapping + [-1] * pad_size

        windows_ids.append(prepared)
        windows_mask.append(attention_mask)
        windows_to_global.append(mapping)
        if end >= len(token_ids):
            break
        start += window_stride
    return windows_ids, windows_mask, windows_to_global


def map_edus_to_subwords(
    offsets: list[tuple[int, int]],
    edu_char_spans: list[tuple[int, int]],
) -> list[list[int]]:
    edu_to_subwords: list[list[int]] = []
    for char_start, char_end in edu_char_spans:
        indices = [
            index
            for index, (subword_start, subword_end) in enumerate(offsets)
            if subword_end > char_start and subword_start < char_end
        ]
        if not indices and offsets:
            center = (char_start + char_end) / 2.0
            nearest = min(
                range(len(offsets)),
                key=lambda idx: abs(((offsets[idx][0] + offsets[idx][1]) / 2.0) - center),
            )
            indices = [nearest]
        edu_to_subwords.append(indices)
    return edu_to_subwords


def prepare_document_features(
    text: str,
    graph: dict[str, Any],
    tokenizer,
    max_length: int,
    window_stride: int,
    representation_text: str | None = None,
) -> dict[str, Any]:
    normalized_text = normalize_text(representation_text if representation_text is not None else text)
    parser_tokens = [str(token) for token in graph["parser_tokens"]]
    token_spans = align_tokens_to_text(normalized_text, parser_tokens)

    encoded = tokenizer(
        normalized_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    token_ids = list(encoded["input_ids"])
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    window_input_ids, window_attention_mask, window_to_global = build_windows(
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_length=max_length,
        window_stride=window_stride,
    )

    doc_encoding = tokenizer(
        normalized_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )

    edu_char_spans: list[tuple[int, int]] = []
    for token_start, token_end in graph["edu_token_spans"]:
        char_start = token_spans[token_start][0]
        char_end = token_spans[token_end][1]
        edu_char_spans.append((char_start, char_end))
    edu_to_subwords = map_edus_to_subwords(offsets=offsets, edu_char_spans=edu_char_spans)

    return {
        "doc_input_ids": list(doc_encoding["input_ids"]),
        "doc_attention_mask": list(doc_encoding["attention_mask"]),
        "window_input_ids": window_input_ids,
        "window_attention_mask": window_attention_mask,
        "window_to_global": window_to_global,
        "edu_to_subwords": edu_to_subwords,
    }


def build_feature_store(
    prefix_root: Path,
    prefix: str,
    graph_cache: dict[str, Any],
    tokenizer,
    max_length: int,
    window_stride: int,
    representation_texts: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    feature_store: dict[str, dict[str, Any]] = {}
    for _, split_paths in get_fold_paths(prefix_root):
        for split_path in split_paths.values():
            for record in load_jsonl(split_path):
                text_id = str(record["text_id"])
                doc_id = f"{prefix}_{text_id}"
                if doc_id in feature_store:
                    continue
                graph = graph_cache[doc_id]
                encoded = prepare_document_features(
                    text=str(record["text"]),
                    graph=graph,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    window_stride=window_stride,
                    representation_text=representation_texts.get(doc_id) if representation_texts else None,
                )
                feature_store[doc_id] = {
                    "doc_id": doc_id,
                    "text_id": text_id,
                    "graph": graph,
                    "sota_text_available": bool(representation_texts is None or doc_id in representation_texts),
                    **encoded,
                }
    return feature_store


def get_sota_document_text(sents: list[str]) -> str:
    """Match EntyRelCoh/fusion/task_dataset.py::get_sent_boundary text."""
    return " ".join(str(sent) for sent in sents)


def load_sota_texts_for_prefix(sota_data_root: Path, prefix: str) -> dict[str, str]:
    """Load document text exactly as the SOTA text-only dataset builds it."""
    dataset_dir = sota_data_root / f"gcdc_{prefix}"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing SOTA text-only data directory: {dataset_dir}")
    texts: dict[str, str] = {}
    for fold_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()):
        for split in ("train", "dev", "test"):
            split_path = fold_dir / f"{split}.json"
            if not split_path.exists():
                continue
            for record in load_jsonl(split_path):
                text_id = str(record.get("text_id", record.get("id")))
                doc_id = f"{prefix}_{text_id}"
                if "sents" not in record:
                    raise ValueError(f"SOTA text-only record missing `sents`: {split_path}")
                if len(record["sents"]) == 0:
                    # EntyRelCoh's SentDataset skips records with no sentences.
                    # Keep the same sample set when using SOTA-aligned text.
                    continue
                text = get_sota_document_text([str(sent) for sent in record["sents"]])
                previous = texts.get(doc_id)
                if previous is not None and previous != text:
                    raise ValueError(f"Inconsistent SOTA text for {doc_id} across folds/splits.")
                texts[doc_id] = text
    if not texts:
        raise ValueError(f"No SOTA text records found under {dataset_dir}")
    return texts


def _sota_cache_filename(model_name: str, max_length: int) -> str:
    model_tag = sanitize_path_component(model_name)
    return f"avg_sota_text_roberta_cached_{model_tag}_{max_length}_v3.pt"


def _sota_pickle_model_tag(model_name: str) -> str:
    lower = model_name.lower()
    if "roberta" in lower:
        return "roberta"
    if "bert" in lower:
        return "bert"
    return sanitize_path_component(model_name)


def _load_sota_pickle_doc_vectors(
    sota_data_root: Path | None,
    prefix: str,
    encoder_name: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Load document vectors produced by the original SOTA text-only cache."""
    if sota_data_root is None:
        return {}
    dataset_dir = sota_data_root / f"gcdc_{prefix}"
    if not dataset_dir.exists():
        return {}
    model_tag = _sota_pickle_model_tag(encoder_name)
    vectors: dict[str, torch.Tensor] = {}
    for fold_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()):
        fusion_dir = fold_dir / "fusion_vectors"
        if not fusion_dir.exists():
            continue
        for split in ("train", "dev", "test"):
            vector_path = fusion_dir / f"avg_{split}_{model_tag}_{max_length}-15.pkl"
            if not vector_path.exists():
                continue
            with vector_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, list) or len(payload) < 9:
                continue
            doc_vectors, text_ids = payload[0], payload[8]
            for row, text_id in enumerate(text_ids):
                doc_id = f"{prefix}_{text_id}"
                doc_vector = doc_vectors[row].detach().cpu().contiguous()
                previous = vectors.get(doc_id)
                if previous is not None and not torch.allclose(previous, doc_vector, atol=1.0e-6, rtol=1.0e-6):
                    raise ValueError(f"Inconsistent SOTA cached doc vector for {doc_id}: {vector_path}")
                vectors[doc_id] = doc_vector
    return vectors


def attach_sota_cached_representations(
    feature_store: dict[str, dict[str, Any]],
    encoder_name: str,
    tokenizer,
    cache_dir: Path,
    prefix: str,
    max_length: int,
    device: torch.device,
    batch_size: int = 8,
    rebuild: bool = False,
    sota_data_root: Path | None = None,
) -> None:
    """
    Attach frozen RoBERTa representations using the same cache pattern as the
    SOTA text-only implementation.

    When the original SOTA pkl cache exists, the document vector is loaded from
    that cache directly.  Otherwise it is recomputed as mask-aware mean pooling
    over RoBERTa last hidden states, including special tokens, exactly as in
    `models/entyrelcoh/task_dataset.py`.  For RST spans we cache
    the content-token states from the same truncated document pass.  Span
    indices that fall beyond the truncation boundary are ignored downstream.
    """
    cache_root = cache_dir / prefix / "fusion_vectors"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _sota_cache_filename(encoder_name, max_length)
    expected_doc_ids = sorted(feature_store)
    payload: dict[str, Any] | None = None
    if cache_path.exists() and not rebuild:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            payload = None
    if (
        payload is None
        or int(payload.get("cache_version", 0)) != 3
        or payload.get("encoder_name") != encoder_name
        or int(payload.get("max_length", 0)) != int(max_length)
        or sorted(payload.get("representations", {})) != expected_doc_ids
    ):
        sota_doc_vectors = _load_sota_pickle_doc_vectors(
            sota_data_root=sota_data_root,
            prefix=prefix,
            encoder_name=encoder_name,
            max_length=max_length,
        )
        encoder = AutoModel.from_pretrained(encoder_name).to(device)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        representations: dict[str, dict[str, torch.Tensor]] = {}
        doc_ids = expected_doc_ids
        steps = len(doc_ids) // max(int(batch_size), 1) + 1
        for step in trange(steps, desc=f"cache {prefix} frozen RoBERTa states", leave=False):
            start = step * max(int(batch_size), 1)
            end = min((step + 1) * max(int(batch_size), 1), len(doc_ids))
            if start >= end:
                break
            batch_doc_ids = doc_ids[start:end]
            input_ids = torch.tensor(
                [feature_store[doc_id]["doc_input_ids"] for doc_id in batch_doc_ids],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.tensor(
                [feature_store[doc_id]["doc_attention_mask"] for doc_id in batch_doc_ids],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                sequence_states = outputs.last_hidden_state
                lengths = attention_mask.sum(dim=-1).clamp_min(1).to(dtype=sequence_states.dtype)
                doc_repr = (sequence_states * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
            for row, doc_id in enumerate(batch_doc_ids):
                valid_length = int(attention_mask[row].sum().item())
                content_length = max(valid_length - tokenizer.num_special_tokens_to_add(pair=False), 0)
                # RoBERTa stores the first content subword at position 1.
                content_states = sequence_states[row, 1 : 1 + content_length].detach().cpu().contiguous()
                cached_doc_repr = sota_doc_vectors.get(doc_id)
                representations[doc_id] = {
                    "doc_repr": (
                        cached_doc_repr
                        if cached_doc_repr is not None
                        else doc_repr[row].detach().cpu().contiguous()
                    ),
                    "global_states": content_states,
                }
        payload = {
            "cache_version": 3,
            "encoder_name": encoder_name,
            "pooler_type": "avg",
            "text_source": "sota_get_sent_boundary_join_sents",
            "doc_repr_source": "sota_pickle_if_available_else_recomputed",
            "sota_pickle_doc_vectors": len(sota_doc_vectors),
            "max_length": int(max_length),
            "representations": representations,
        }
        torch.save(payload, cache_path)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    representations = payload["representations"]
    for doc_id, item in feature_store.items():
        cached = representations[doc_id]
        item["doc_repr"] = cached["doc_repr"]
        item["global_states"] = cached["global_states"]


def _span_mean(states: torch.Tensor, indices: list[int]) -> torch.Tensor:
    valid = [idx for idx in indices if 0 <= int(idx) < int(states.shape[0])]
    if not valid:
        return torch.zeros(states.shape[-1], dtype=states.dtype)
    return states[torch.tensor(valid, dtype=torch.long)].mean(dim=0)


def _tree_metadata_for_cache(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph["nodes"]
    num_nodes = len(nodes)
    key_to_index = {(str(node["kind"]), int(node["parser_id"])): int(node["index"]) for node in nodes}
    parent = [-1] * num_nodes
    branch_from_parent = [0] * num_nodes
    nuc_from_parent = [0] * num_nodes

    def edge_nuc_role(parent_node: dict[str, Any], side: str) -> int:
        label = str(parent_node.get("nuclearity_label", "") or "")
        if label == "NS":
            return 1 if side == "left" else 2
        if label == "SN":
            return 2 if side == "left" else 1
        if label == "NN":
            return 3
        return 0

    for parent_idx in graph["internal_indices"]:
        parent_node = nodes[parent_idx]
        for side_name, child_field, branch_id in (("left", "left_child", 0), ("right", "right_child", 1)):
            child = parent_node[child_field]
            child_idx = key_to_index[(str(child["kind"]), int(child["id"]))]
            parent[child_idx] = int(parent_idx)
            branch_from_parent[child_idx] = branch_id
            nuc_from_parent[child_idx] = edge_nuc_role(parent_node, side_name)

    ancestor_chains: list[list[int]] = []
    branch_paths: list[list[int]] = []
    nuc_paths: list[list[int]] = []
    for node_idx in range(num_nodes):
        chain = []
        branch_steps = []
        nuc_steps = []
        current = node_idx
        while current >= 0:
            chain.append(current)
            if parent[current] >= 0:
                branch_steps.append(branch_from_parent[current])
                nuc_steps.append(nuc_from_parent[current])
            current = parent[current]
        ancestor_chains.append(chain)
        branch_paths.append(list(reversed(branch_steps)))
        nuc_paths.append(list(reversed(nuc_steps)))

    depths = [int(node["depth"]) for node in nodes]
    dist = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    ancestor_relation = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    same_subtree = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    ancestor_sets = [set(chain) for chain in ancestor_chains]
    root_index = int(graph["root_index"])
    root_child = [-1] * num_nodes
    for node_idx in range(num_nodes):
        if node_idx == root_index:
            continue
        current = node_idx
        while parent[current] >= 0 and parent[current] != root_index:
            current = parent[current]
        root_child[node_idx] = current if parent[current] == root_index else -1

    for i in range(num_nodes):
        for j in range(num_nodes):
            common = ancestor_sets[i].intersection(ancestor_sets[j])
            lca = max(common, key=lambda idx: depths[idx]) if common else root_index
            dist[i, j] = max(depths[i] + depths[j] - 2 * depths[lca], 0)
            if i in ancestor_sets[j] and i != j:
                ancestor_relation[i, j] = 1
            elif j in ancestor_sets[i] and i != j:
                ancestor_relation[i, j] = -1
            if i == j or (root_child[i] >= 0 and root_child[i] == root_child[j]):
                same_subtree[i, j] = 1
    return {
        "tree_dist": dist,
        "ancestor_relation": ancestor_relation,
        "same_subtree": same_subtree,
        "branch_paths": branch_paths,
        "nuc_paths": nuc_paths,
    }


def attach_rst_tensor_cache(
    feature_store: dict[str, dict[str, Any]],
    cache_dir: Path,
    prefix: str,
    encoder_name: str,
    max_length: int,
    rebuild: bool = False,
) -> None:
    cache_root = cache_dir / prefix
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"rst_tensors_{sanitize_path_component(encoder_name)}_{max_length}.pt"
    expected_doc_ids = sorted(feature_store)
    payload: dict[str, Any] | None = None
    if cache_path.exists() and not rebuild:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            payload = None
    if (
        payload is None
        or int(payload.get("cache_version", 0)) != 1
        or payload.get("encoder_name") != encoder_name
        or int(payload.get("max_length", 0)) != int(max_length)
        or sorted(payload.get("documents", {})) != expected_doc_ids
    ):
        documents: dict[str, dict[str, Any]] = {}
        for doc_id in expected_doc_ids:
            item = feature_store[doc_id]
            graph = item["graph"]
            states = item["global_states"]
            edu_to_subwords = item["edu_to_subwords"]
            edu_states = []
            for edu_index in range(int(graph["num_edus"])):
                edu_states.append(_span_mean(states, edu_to_subwords[edu_index]))
            span_states = []
            for node in graph["nodes"]:
                indices: list[int] = []
                for edu_index in range(int(node["edu_start"]), int(node["edu_end"]) + 1):
                    indices.extend(edu_to_subwords[edu_index])
                span_states.append(_span_mean(states, indices))
            documents[doc_id] = {
                "edu_states": torch.stack(edu_states, dim=0),
                "span_states": torch.stack(span_states, dim=0),
                **_tree_metadata_for_cache(graph),
            }
        payload = {
            "cache_version": 1,
            "encoder_name": encoder_name,
            "max_length": int(max_length),
            "documents": documents,
        }
        torch.save(payload, cache_path)

    documents = payload["documents"]
    for doc_id, item in feature_store.items():
        item["rst_tensors"] = documents[doc_id]


def label_sort_key(label: str) -> tuple[int, float | str]:
    try:
        return 0, float(label)
    except ValueError:
        return 1, label


def build_label_mapping(*split_records: list[dict[str, Any]]) -> dict[str, int]:
    labels = sorted({str(record["score"]) for records in split_records for record in records}, key=label_sort_key)
    return {label: index for index, label in enumerate(labels)}


def build_examples_for_split(
    split_path: Path,
    prefix: str,
    feature_store: dict[str, dict[str, Any]],
    label_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in load_jsonl(split_path):
        text_id = str(record["text_id"])
        doc_id = f"{prefix}_{text_id}"
        base = feature_store[doc_id]
        examples.append(
            {
                **base,
                "label": label_to_id[str(record["score"])],
            }
        )
    return examples


class PreparedDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class BatchCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> DocumentBatch:
        max_windows = max(len(item["window_input_ids"]) for item in batch)
        max_window_length = max(len(window) for item in batch for window in item["window_input_ids"])
        max_edus = max(len(item["edu_to_subwords"]) for item in batch)
        max_edu_width = max((len(indices) for item in batch for indices in item["edu_to_subwords"]), default=1)

        doc_length = len(batch[0]["doc_input_ids"])
        doc_input_ids = torch.zeros(len(batch), doc_length, dtype=torch.long)
        doc_attention_mask = torch.zeros(len(batch), doc_length, dtype=torch.long)
        window_input_ids = torch.zeros(len(batch), max_windows, max_window_length, dtype=torch.long)
        window_attention_mask = torch.zeros(len(batch), max_windows, max_window_length, dtype=torch.long)
        window_to_global = torch.full((len(batch), max_windows, max_window_length), -1, dtype=torch.long)
        edu_to_subwords = torch.zeros(len(batch), max_edus, max_edu_width, dtype=torch.long)
        edu_mask = torch.zeros(len(batch), max_edus, max_edu_width, dtype=torch.bool)
        edu_valid_mask = torch.zeros(len(batch), max_edus, dtype=torch.bool)
        labels = torch.zeros(len(batch), dtype=torch.long)
        q_emp = torch.zeros(len(batch), 3, dtype=torch.float)
        mu_ratings = torch.zeros(len(batch), dtype=torch.float)
        au_coh = torch.zeros(len(batch), dtype=torch.float)
        raw_score_5 = torch.zeros(len(batch), dtype=torch.long)
        has_cached_repr = "doc_repr" in batch[0] and "global_states" in batch[0]
        if has_cached_repr:
            hidden_size = int(batch[0]["doc_repr"].shape[-1])
            max_global_length = max(int(item["global_states"].shape[0]) for item in batch)
            doc_repr = torch.zeros(len(batch), hidden_size, dtype=torch.float)
            global_states = torch.zeros(len(batch), max_global_length, hidden_size, dtype=torch.float)
            global_state_mask = torch.zeros(len(batch), max_global_length, dtype=torch.bool)
        else:
            doc_repr = torch.zeros(len(batch), 0, dtype=torch.float)
            global_states = torch.zeros(len(batch), 0, 0, dtype=torch.float)
            global_state_mask = torch.zeros(len(batch), 0, dtype=torch.bool)

        graphs: list[dict[str, Any]] = []
        rst_tensors: list[dict[str, Any] | None] = []
        doc_ids: list[str] = []
        text_ids: list[str] = []
        for row, item in enumerate(batch):
            doc_input_ids[row] = torch.tensor(item["doc_input_ids"], dtype=torch.long)
            doc_attention_mask[row] = torch.tensor(item["doc_attention_mask"], dtype=torch.long)
            labels[row] = int(item["label"])
            q_emp[row] = torch.tensor(item["q_emp"], dtype=torch.float)
            mu_ratings[row] = float(item["mu_rating"])
            au_coh[row] = float(item["au_coh"])
            raw_score_5[row] = int(item.get("raw_score_5", 0))
            graphs.append(item["graph"])
            rst_tensors.append(item.get("rst_tensors"))
            doc_ids.append(str(item["doc_id"]))
            text_ids.append(str(item["text_id"]))
            if has_cached_repr:
                cached_doc = item["doc_repr"].to(dtype=torch.float)
                cached_global = item["global_states"].to(dtype=torch.float)
                doc_repr[row] = cached_doc
                global_states[row, : cached_global.shape[0]] = cached_global
                global_state_mask[row, : cached_global.shape[0]] = True

            for window_index, ids in enumerate(item["window_input_ids"]):
                ids_tensor = torch.tensor(ids, dtype=torch.long)
                mask_tensor = torch.tensor(item["window_attention_mask"][window_index], dtype=torch.long)
                global_tensor = torch.tensor(item["window_to_global"][window_index], dtype=torch.long)
                window_input_ids[row, window_index, : len(ids)] = ids_tensor
                window_attention_mask[row, window_index, : len(ids)] = mask_tensor
                window_to_global[row, window_index, : len(ids)] = global_tensor

            for edu_index, subword_indices in enumerate(item["edu_to_subwords"]):
                if subword_indices:
                    edu_to_subwords[row, edu_index, : len(subword_indices)] = torch.tensor(
                        subword_indices,
                        dtype=torch.long,
                    )
                    edu_mask[row, edu_index, : len(subword_indices)] = True
                edu_valid_mask[row, edu_index] = True

        return DocumentBatch(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            window_input_ids=window_input_ids,
            window_attention_mask=window_attention_mask,
            window_to_global=window_to_global,
            doc_repr=doc_repr,
            global_states=global_states,
            global_state_mask=global_state_mask,
            rst_tensors=rst_tensors,
            edu_to_subwords=edu_to_subwords,
            edu_mask=edu_mask,
            edu_valid_mask=edu_valid_mask,
            labels=labels,
            q_emp=q_emp,
            mu_ratings=mu_ratings,
            au_coh=au_coh,
            raw_score_5=raw_score_5,
            graphs=graphs,
            doc_ids=doc_ids,
            text_ids=text_ids,
        )
