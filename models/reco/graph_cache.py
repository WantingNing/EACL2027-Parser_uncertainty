from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import torch


PREFIXES = ("clinton", "enron", "yahoo", "yelp", "cohe")
CACHE_VERSION = 2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_split_root(data_root: Path, prefix: str) -> Path:
    gcdc_split_root = data_root / f"gcdc_{prefix}"
    if gcdc_split_root.exists():
        return gcdc_split_root
    generic_split_root = data_root / prefix
    if generic_split_root.exists():
        return generic_split_root
    raise FileNotFoundError(
        f"Missing split directory for {prefix}: tried {gcdc_split_root} and {generic_split_root}"
    )


def get_fold_paths(prefix_root: Path) -> list[tuple[int, dict[str, Path]]]:
    fold_paths: list[tuple[int, dict[str, Path]]] = []
    for fold_dir in sorted(path for path in prefix_root.iterdir() if path.is_dir() and path.name.isdigit()):
        fold_paths.append(
            (
                int(fold_dir.name),
                {
                    "train": fold_dir / "train.json",
                    "dev": fold_dir / "dev.json",
                    "test": fold_dir / "test.json",
                },
            )
        )
    if not fold_paths:
        raise ValueError(f"No fold directories found under {prefix_root}.")
    return fold_paths


def collect_split_records(prefix_root: Path) -> dict[str, dict[str, Any]]:
    records_by_doc: dict[str, dict[str, Any]] = {}
    prefix = prefix_root.name.replace("gcdc_", "")
    for _, split_paths in get_fold_paths(prefix_root):
        for split_name, split_path in split_paths.items():
            for record in load_jsonl(split_path):
                text_id = str(record["text_id"])
                doc_id = f"{prefix}_{text_id}"
                if doc_id not in records_by_doc:
                    records_by_doc[doc_id] = {
                        "doc_id": doc_id,
                        "text_id": text_id,
                        "text": str(record["text"]),
                        "score": str(record["score"]),
                        "seen_in": {split_name},
                    }
                else:
                    records_by_doc[doc_id]["seen_in"].add(split_name)
    return records_by_doc


def parser_path_for(parser_root: Path, prefix: str, doc_id: str) -> Path:
    if prefix == "cohe":
        return parser_root / "cohesentia" / prefix / f"{doc_id}.json"
    return parser_root / "gcdc" / prefix / f"{doc_id}.json"


def child_key(child: dict[str, Any]) -> tuple[str, int]:
    return str(child["kind"]), int(child["id"])


def normalized_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    positive = [value for value in probabilities if value > 0.0]
    if not positive:
        return 0.0
    raw = -sum(value * math.log(value) for value in positive)
    denom = math.log(len(probabilities))
    if denom <= 0.0:
        return 0.0
    return raw / denom


def relation_confidence_margin(probabilities: list[float]) -> tuple[float, float, float]:
    if not probabilities:
        return 0.0, 0.0, 0.0
    ranked = sorted((float(value) for value in probabilities), reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else 0.0
    ambiguity = (1.0 - (top1 - top2)) * (top1 + top2)
    return top1, top1 - top2, ambiguity


def compute_depths(node_map: dict[int, dict[str, Any]], root_node_id: int) -> tuple[dict[tuple[str, int], int], int]:
    depths: dict[tuple[str, int], int] = {("internal", root_node_id): 0}
    queue: deque[tuple[str, int]] = deque([("internal", root_node_id)])
    max_depth = 0
    while queue:
        kind, ident = queue.popleft()
        depth = depths[(kind, ident)]
        max_depth = max(max_depth, depth)
        if kind != "internal":
            continue
        node = node_map[ident]
        for field in ("left_child", "right_child"):
            child = node[field]
            key = child_key(child)
            if key not in depths:
                depths[key] = depth + 1
                queue.append(key)
    return depths, max_depth


def compute_root_path(node_map: dict[int, dict[str, Any]], root_node_id: int) -> set[tuple[str, int]]:
    deepest_cache: dict[tuple[str, int], int] = {}

    def deepest_depth(key: tuple[str, int]) -> int:
        if key in deepest_cache:
            return deepest_cache[key]
        kind, ident = key
        if kind == "leaf":
            deepest_cache[key] = 0
            return 0
        node = node_map[ident]
        left_depth = deepest_depth(child_key(node["left_child"]))
        right_depth = deepest_depth(child_key(node["right_child"]))
        deepest_cache[key] = 1 + max(left_depth, right_depth)
        return deepest_cache[key]

    path_keys: set[tuple[str, int]] = set()
    current = ("internal", root_node_id)
    path_keys.add(current)
    while current[0] == "internal":
        node = node_map[current[1]]
        left_key = child_key(node["left_child"])
        right_key = child_key(node["right_child"])
        left_depth = deepest_depth(left_key)
        right_depth = deepest_depth(right_key)
        next_key = left_key if left_depth >= right_depth else right_key
        path_keys.add(next_key)
        current = next_key
    return path_keys


def edge_side(parent_node: dict[str, Any], child: dict[str, Any]) -> str:
    if int(parent_node["left_child"]["id"]) == int(child["id"]) and str(parent_node["left_child"]["kind"]) == str(child["kind"]):
        return "left"
    if int(parent_node["right_child"]["id"]) == int(child["id"]) and str(parent_node["right_child"]["kind"]) == str(child["kind"]):
        return "right"
    return "none"


def edge_nuclearity(parent_node: dict[str, Any], child: dict[str, Any]) -> str:
    label = str(parent_node.get("nuclearity_label", "") or "")
    side = edge_side(parent_node, child)
    if label == "NS":
        return "N" if side == "left" else "S" if side == "right" else "none"
    if label == "SN":
        return "S" if side == "left" else "N" if side == "right" else "none"
    if label == "NN":
        return "NN"
    return "none"


def root_gap_bucket(src_depth: int, dst_depth: int) -> str:
    gap = abs(int(src_depth) - int(dst_depth))
    gap = max(1, min(gap, 4))
    if gap == 1:
        return "root_gap_1"
    if gap == 2:
        return "root_gap_2"
    if gap == 3:
        return "root_gap_3"
    return "root_gap_4plus"


def distribution_proxy(probabilities: list[float]) -> tuple[float, float, float, float]:
    probs = [float(value) for value in probabilities]
    entropy = normalized_entropy(probs)
    confidence, margin, ambiguity = relation_confidence_margin(probs)
    return entropy, confidence, margin, ambiguity


def relation_proxy(node: dict[str, Any]) -> tuple[float, float, float, float]:
    return distribution_proxy([float(value) for value in node.get("relation_distribution", [])])


def build_doc_graph(payload: dict[str, Any], parser_lambda: float, parser_beta: list[float]) -> dict[str, Any]:
    tree = payload["tree"]
    root_node_id = int(tree["root_node_id"])
    internal_nodes = sorted(tree["nodes"], key=lambda node: (int(node["span"]["edu_start"]), int(node["node_id"])))
    node_map = {int(node["node_id"]): node for node in internal_nodes}

    num_edus = len(payload["edus"])
    depths, max_depth = compute_depths(node_map=node_map, root_node_id=root_node_id)
    root_path_keys = compute_root_path(node_map=node_map, root_node_id=root_node_id)

    edus_by_id = {int(edu["edu_id"]): edu for edu in payload["edus"]}
    dense_index: dict[tuple[str, int], int] = {}
    nodes: list[dict[str, Any]] = []

    for edu in sorted(payload["edus"], key=lambda item: int(item["edu_id"])):
        edu_id = int(edu["edu_id"])
        key = ("leaf", edu_id)
        depth = depths.get(key, max_depth)
        dense_index[key] = len(nodes)
        nodes.append(
            {
                "index": len(nodes),
                "kind": "leaf",
                "parser_id": edu_id,
                "edu_start": edu_id,
                "edu_end": edu_id,
                "token_start": int(edu["token_start"]),
                "token_end": int(edu["token_end"]),
                "depth": depth,
                "span_len": 1,
                "is_root": 0,
                "is_root_path": 1 if key in root_path_keys else 0,
                "node_type": "leaf",
                "relation_label": None,
                "nuclearity_label": None,
                "relation_distribution": [0.0] * 18,
                "relation_entropy_norm": 0.0,
                "nuclearity_entropy_norm": 0.0,
                "split_entropy_norm": 0.0,
                "entropy_norm": 0.0,
                "confidence": 0.0,
                "margin": 0.0,
                "ambiguity": 0.0,
                "relation_rri": 0.0,
                "nuclearity_rri": 0.0,
                "split_rri": 0.0,
                "rri": 0.0,
                "parser_eu": 0.0,
                "topology": [
                    float(depth),
                    1.0,
                    0.0,
                    float(1 if key in root_path_keys else 0),
                    float(depth / max(max_depth, 1)),
                ],
            }
        )

    internal_indices: list[int] = []
    for node in internal_nodes:
        node_id = int(node["node_id"])
        key = ("internal", node_id)
        depth = depths[key]
        edu_start = int(node["span"]["edu_start"])
        edu_end = int(node["span"]["edu_end"])
        span_len = edu_end - edu_start + 1
        probs = [float(value) for value in node.get("relation_distribution", [])]
        relation_entropy_norm, confidence, margin, ambiguity = relation_proxy(node)
        nuclearity_entropy_norm, _, _, _ = distribution_proxy(
            [float(value) for value in node.get("nuclearity_distribution", [])]
        )
        split_entropy_norm, _, _, _ = distribution_proxy([float(value) for value in node.get("split_distribution", [])])
        relation_rri = relation_entropy_norm * math.log1p(span_len) * math.exp(-float(parser_lambda) * depth)
        nuclearity_rri = nuclearity_entropy_norm * math.log1p(span_len) * math.exp(-float(parser_lambda) * depth)
        split_rri = split_entropy_norm * math.log1p(span_len) * math.exp(-float(parser_lambda) * depth)
        parser_eu = (
            float(parser_beta[0]) * relation_entropy_norm
            + float(parser_beta[1]) * (1.0 - confidence)
            + float(parser_beta[2]) * (1.0 - margin)
            + float(parser_beta[3]) * ambiguity
            + float(parser_beta[4]) * relation_rri
        )
        left_leaf = edus_by_id[edu_start]
        right_leaf = edus_by_id[edu_end]
        dense_index[key] = len(nodes)
        nodes.append(
            {
                "index": len(nodes),
                "kind": "internal",
                "parser_id": node_id,
                "edu_start": edu_start,
                "edu_end": edu_end,
                "token_start": int(left_leaf["token_start"]),
                "token_end": int(right_leaf["token_end"]),
                "depth": depth,
                "span_len": span_len,
                "is_root": 1 if node_id == root_node_id else 0,
                "is_root_path": 1 if key in root_path_keys else 0,
                "node_type": "internal",
                "relation_label": str(node.get("relation_label", "") or "") or None,
                "nuclearity_label": str(node.get("nuclearity_label", "") or "") or None,
                "relation_distribution": probs,
                "relation_entropy_norm": relation_entropy_norm,
                "nuclearity_entropy_norm": nuclearity_entropy_norm,
                "split_entropy_norm": split_entropy_norm,
                "entropy_norm": relation_entropy_norm,
                "confidence": confidence,
                "margin": margin,
                "ambiguity": ambiguity,
                "relation_rri": relation_rri,
                "nuclearity_rri": nuclearity_rri,
                "split_rri": split_rri,
                "rri": relation_rri,
                "parser_eu": parser_eu,
                "topology": [
                    float(depth),
                    float(span_len),
                    float(1 if node_id == root_node_id else 0),
                    float(1 if key in root_path_keys else 0),
                    float(depth / max(max_depth, 1)),
                ],
                "left_child": {
                    "kind": str(node["left_child"]["kind"]),
                    "id": int(node["left_child"]["id"]),
                },
                "right_child": {
                    "kind": str(node["right_child"]["kind"]),
                    "id": int(node["right_child"]["id"]),
                },
            }
        )
        internal_indices.append(len(nodes) - 1)

    edges: list[dict[str, Any]] = []

    def add_edge(
        src_key: tuple[str, int],
        dst_key: tuple[str, int],
        direction: str,
        side: str,
        nuclearity_role: str,
        relation_label: str,
        distance_bucket: str,
    ) -> None:
        src_index = dense_index[src_key]
        dst_index = dense_index[dst_key]
        edges.append(
            {
                "src": src_index,
                "dst": dst_index,
                "dir": direction,
                "side": side,
                "nuc": nuclearity_role,
                "rel": relation_label,
                "dist": distance_bucket,
                "kind": direction,
            }
        )

    for node in internal_nodes:
        parent_id = int(node["node_id"])
        parent_key = ("internal", parent_id)
        left_child = node["left_child"]
        right_child = node["right_child"]
        left_key = child_key(left_child)
        right_key = child_key(right_child)
        parent_relation = str(node.get("relation_label", "") or "") or "<UNK>"

        for child in (left_child, right_child):
            child_k = child_key(child)
            add_edge(
                src_key=parent_key,
                dst_key=child_k,
                direction="parent_to_child",
                side=edge_side(node, child),
                nuclearity_role=edge_nuclearity(node, child),
                relation_label=parent_relation,
                distance_bucket="pc_1hop",
            )
            add_edge(
                src_key=child_k,
                dst_key=parent_key,
                direction="child_to_parent",
                side=edge_side(node, child),
                nuclearity_role=edge_nuclearity(node, child),
                relation_label=parent_relation,
                distance_bucket="pc_1hop",
            )

        add_edge(
            src_key=left_key,
            dst_key=right_key,
            direction="sibling",
            side="left_to_right",
            nuclearity_role="none",
            relation_label=parent_relation,
            distance_bucket="same_parent",
        )
        add_edge(
            src_key=right_key,
            dst_key=left_key,
            direction="sibling",
            side="right_to_left",
            nuclearity_role="none",
            relation_label=parent_relation,
            distance_bucket="same_parent",
        )

    root_key = ("internal", root_node_id)
    root_depth = depths[root_key]
    for path_key in sorted(root_path_keys, key=lambda item: (0 if item[0] == "internal" else 1, item[1])):
        if path_key == root_key or path_key[0] != "internal":
            continue
        gap_bucket = root_gap_bucket(depths[path_key], root_depth)
        add_edge(
            src_key=root_key,
            dst_key=path_key,
            direction="root_path",
            side="none",
            nuclearity_role="none",
            relation_label="root_path",
            distance_bucket=gap_bucket,
        )
        add_edge(
            src_key=path_key,
            dst_key=root_key,
            direction="root_path",
            side="none",
            nuclearity_role="none",
            relation_label="root_path",
            distance_bucket=gap_bucket,
        )

    edu_token_spans = [
        (int(edu["token_start"]), int(edu["token_end"]))
        for edu in sorted(payload["edus"], key=lambda item: int(item["edu_id"]))
    ]
    return {
        "doc_id": str(payload["doc_id"]),
        "parser_tokens": [str(token) for token in payload["tokens"]],
        "edu_token_spans": edu_token_spans,
        "num_edus": num_edus,
        "root_node_id": root_node_id,
        "root_index": dense_index[root_key],
        "nodes": nodes,
        "internal_indices": internal_indices,
        "edges": edges,
        "max_depth": max_depth,
    }


def build_prefix_graph_cache(
    prefix: str,
    data_root: Path,
    parser_root: Path,
    cache_dir: Path,
    parser_lambda: float,
    parser_beta: list[float],
) -> Path:
    prefix_root = get_split_root(data_root=data_root, prefix=prefix)
    records_by_doc = collect_split_records(prefix_root)
    graph_by_doc: dict[str, dict[str, Any]] = {}
    for doc_id in sorted(records_by_doc):
        parser_path = parser_path_for(parser_root=parser_root, prefix=prefix, doc_id=doc_id)
        if not parser_path.exists():
            raise FileNotFoundError(f"Missing parser file for {doc_id}: {parser_path}")
        payload = json.loads(parser_path.read_text(encoding="utf-8"))
        graph_by_doc[doc_id] = build_doc_graph(
            payload=payload,
            parser_lambda=parser_lambda,
            parser_beta=parser_beta,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{prefix}_graphs.pt"
    torch.save(
        {
            "prefix": prefix,
            "cache_version": CACHE_VERSION,
            "graphs": graph_by_doc,
            "doc_ids": sorted(graph_by_doc),
            "parser_lambda": float(parser_lambda),
            "parser_beta": [float(value) for value in parser_beta],
        },
        cache_path,
    )
    return cache_path


def load_or_build_graph_cache(
    prefix: str,
    data_root: Path,
    parser_root: Path,
    cache_dir: Path,
    parser_lambda: float,
    parser_beta: list[float],
) -> dict[str, Any]:
    cache_path = cache_dir / f"{prefix}_graphs.pt"
    rebuild = not cache_path.exists()
    if not rebuild:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                int(payload.get("cache_version", 0)) != CACHE_VERSION
                or float(payload.get("parser_lambda", -1.0)) != float(parser_lambda)
                or [float(value) for value in payload.get("parser_beta", [])] != [float(value) for value in parser_beta]
            ):
                rebuild = True
        except Exception:
            rebuild = True
    if rebuild:
        build_prefix_graph_cache(
            prefix=prefix,
            data_root=data_root,
            parser_root=parser_root,
            cache_dir=cache_dir,
            parser_lambda=parser_lambda,
            parser_beta=parser_beta,
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict):
        raise ValueError(f"Malformed graph cache: {cache_path}")
    return graphs
