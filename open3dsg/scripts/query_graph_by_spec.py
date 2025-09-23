# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Query a scene graph for relation triples defined in a JSON spec."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _to_label_set(value: Any) -> set[str]:
    """Return a case-folded set of labels encoded by ``value``."""

    if value is None:
        return set()

    if isinstance(value, str):
        label = value.strip()
        return {label.casefold()} if label else set()

    if isinstance(value, Mapping):
        collected: set[str] = set()
        for key in ("label", "name", "predicate", "relation", "synset", "category", "type"):
            if key not in value:
                continue
            collected.update(_to_label_set(value[key]))
        if collected:
            return collected
        for candidate in value.values():
            collected.update(_to_label_set(candidate))
        return collected

    if isinstance(value, Iterable):
        labels: set[str] = set()
        for item in value:
            labels.update(_to_label_set(item))
        return labels

    return _to_label_set(str(value))


def _label_lookup(mapping: Any, *, node: bool) -> dict[int, str]:
    """Build a mapping from ids to labels for nodes or relations."""

    if mapping is None:
        return {}

    if isinstance(mapping, Mapping):
        branch_key = "objects" if node else "relationships"
        alt_key = "nodes" if node else "predicates"
        if branch_key in mapping:
            return _label_lookup(mapping[branch_key], node=node)
        if alt_key in mapping:
            return _label_lookup(mapping[alt_key], node=node)

        resolved: dict[int, str] = {}
        for raw_key, value in mapping.items():
            try:
                idx = int(raw_key)
            except (TypeError, ValueError):
                continue
            labels = _to_label_set(value)
            if labels:
                resolved[idx] = next(iter(labels))
        return resolved

    if isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
        resolved: dict[int, str] = {}
        for idx, value in enumerate(mapping):
            labels = _to_label_set(value)
            if labels:
                resolved[idx] = next(iter(labels))
        return resolved

    labels = _to_label_set(mapping)
    return {0: next(iter(labels))} if labels else {}


def _iter_edges(edges: Any) -> list[tuple[int, int]]:
    if edges is None:
        return []

    if hasattr(edges, "tolist"):
        edges = edges.tolist()

    result: list[tuple[int, int]] = []
    for entry in edges:
        if entry is None:
            continue
        if hasattr(entry, "tolist"):
            entry = entry.tolist()
        if isinstance(entry, Mapping):
            src = entry.get("source") or entry.get("from")
            dst = entry.get("target") or entry.get("to") or entry.get("end")
            try:
                result.append((int(src), int(dst)))
            except (TypeError, ValueError):
                continue
            continue
        try:
            src, dst = entry[:2]
            result.append((int(src), int(dst)))
        except (TypeError, ValueError, IndexError):
            continue
    return result


def _normalise_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _relation_labels(entry: Any, relation_map: Mapping[int, str]) -> set[str]:
    if entry is None:
        return set()
    if hasattr(entry, "tolist"):
        entry = entry.tolist()

    if isinstance(entry, Mapping):
        return _to_label_set(entry)

    if isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
        values = list(entry)
        if values and all(isinstance(val, (int, float)) for val in values):
            labels = {
                relation_map.get(idx)
                for idx, flag in enumerate(values)
                if bool(flag) and idx in relation_map
            }
            return {label for label in labels if label}
        labels: set[str] = set()
        for item in values:
            labels.update(_relation_labels(item, relation_map))
        return labels

    if isinstance(entry, (int, float)):
        label = relation_map.get(int(entry))
        return {label} if label else set()

    return _to_label_set(entry)


def find_relation_triples(data_dict: Mapping[str, Any], pattern: Mapping[str, Any]) -> list[list[int]]:
    """Return ``[source_id, target_id]`` pairs satisfying the graph pattern."""

    graph_pattern = pattern.get("graph_pattern", []) if pattern else []
    if not graph_pattern:
        return []

    edges = _iter_edges(data_dict.get("edges")) if data_dict else []
    predicate_cat = _normalise_sequence(data_dict.get("predicate_cat")) if data_dict else []
    id2name = data_dict.get("id2name") if data_dict else None

    if not edges or not predicate_cat:
        return []

    node_labels = _label_lookup(id2name, node=True)
    relation_labels = _label_lookup(id2name, node=False)

    matches: list[list[int]] = []
    seen: set[tuple[int, int]] = set()

    for query in graph_pattern:
        if not isinstance(query, Mapping):
            continue
        src_labels = _to_label_set(query.get("source"))
        rel_labels = _to_label_set(query.get("relation"))
        dst_labels = _to_label_set(query.get("end"))
        if not src_labels or not rel_labels or not dst_labels:
            continue

        for idx, (src, dst) in enumerate(edges):
            src_label = node_labels.get(src)
            dst_label = node_labels.get(dst)
            if src_label not in src_labels or dst_label not in dst_labels:
                continue

            rel_entry = predicate_cat[idx] if idx < len(predicate_cat) else None
            rel_values = _relation_labels(rel_entry, relation_labels)
            if rel_labels.isdisjoint(rel_values):
                continue

            pair = (src, dst)
            if pair not in seen:
                matches.append([src, dst])
                seen.add(pair)

    return matches


def _load_graph(path: Path) -> Any:
    if path.suffix.lower() in {".json", ".jsn"}:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Path to the serialized data_dict")
    parser.add_argument("spec", type=Path, help="JSON file describing the query")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dict = _load_graph(args.graph)
    with args.spec.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    matches = find_relation_triples(data_dict, spec)
    print(json.dumps(matches))


if __name__ == "__main__":
    main()

