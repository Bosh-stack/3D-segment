#!/usr/bin/env python3
"""
Query a local Ollama model to return ALL solutions matching a natural-language target
against a 3D scene graph, and SAVE the JSON result to a file.

Scene JSON shape expected:
{
  "objects": [
    { "id": 12, "object_tag": "chair", "bbox_center": [x,y,z], "bbox_extent": [dx,dy,dz] },
    ...
  ],
  "relationships": [
    { "source_id": 12, "target_id": 7, "relation": "next to", "caption": "chair next to desk" },
    ...
  ]
}

Model output (strictly constrained by JSON Schema in `STRUCTURED_SCHEMA`):
{
  "solutions": [
    { "object_ids": [12] },
    {
      "object_ids": [12, 7],
      "edges": [
        { "source_id": 12, "target_id": 7, "relation": "next to", "caption": "chair next to desk" }
      ]
    }
  ]
}
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

try:
    from ollama import chat, Client
except Exception:
    print("Please install the Python client:  pip install ollama", file=sys.stderr)
    raise

# ---- JSON Schema for structured outputs (ALL solutions; no ranking; no 'best') ----
STRUCTURED_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "solutions": {
            "type": "array",
            "description": "All solutions that satisfy the prompt. Order has no meaning.",
            "items": {
                "type": "object",
                "properties": {
                    "object_ids": {
                        "type": "array",
                        "description": "IDs that jointly satisfy the prompt (size 1 for object-only; 2+ for relationship queries).",
                        "minItems": 1,
                        "items": {"type": "integer"}
                    },
                    "edges": {
                        "type": "array",
                        "description": "Supporting relationships among object_ids (omit or empty for object-only).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "integer"},
                                "target_id": {"type": "integer"},
                                "relation": {"type": "string"},
                                "caption":  {"type": "string"}
                            },
                            "required": ["source_id", "target_id", "relation", "caption"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["object_ids"],
                "additionalProperties": False
            }
        }
    },
    "required": ["solutions"],
    "additionalProperties": False
}

SYSTEM_PROMPT = (
    "You will receive a 3D scene graph as JSON with two lists:\n"
    "1) objects: each has id (int), object_tag (string), and optionally bbox_center [x,y,z] and bbox_extent [dx,dy,dz].\n"
    "2) relationships: each has source_id, target_id, relation (string), caption (string).\n\n"
    "The user provides a target expression such as:\n"
    '  - "find a chair"\n'
    '  - "find a chair next to a desk"\n'
    '  - "find a lamp on a table"\n'
    '  - "find a box inside a cabinet"\n\n'
    "Return ALL solutions that satisfy the target:\n"
    "- For object-only targets, each solution is a single id in object_ids.\n"
    "- For relationship targets, each solution is a set of ids that jointly satisfy the relation; include the supporting edges that justify the relation.\n"
    "- Accept common relation synonyms: next to/near/beside; on/on top of; under/below; over/above; in/inside/within; left of/right of; behind/in front of.\n"
    "- Order has no meaning. Do not rank. Do not add scores. Do not include any fields not defined in the schema.\n"
    "- Output MUST be valid JSON matching the schema provided by the caller."
)


def load_scene(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def slim_objects(objs: List[Dict[str, Any]], max_objects: int = 160) -> List[Dict[str, Any]]:
    """Keep only the minimal fields and cap count to avoid long contexts."""
    keep = ("id", "object_tag", "bbox_center", "bbox_extent")
    trimmed = [{k: o[k] for k in keep if k in o} for o in objs]
    return trimmed[:max_objects] if max_objects else trimmed


def call_ollama(scene: Dict[str, Any],
                question: str,
                model: str = "llama3.1:8b",
                host: str = None,
                temperature: float = 0.0,
                max_objects: int = 160,
                use_schema: bool = True) -> Dict[str, Any]:
    """
    Call Ollama /api/chat with structured outputs (JSON Schema via `format`).
    """
    objects = slim_objects(scene.get("objects", []), max_objects=max_objects)
    relationships = scene.get("relationships", [])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": (
            "Objects:\n" + json.dumps(objects, ensure_ascii=False)
            + "\n\nRelationships:\n" + json.dumps(relationships, ensure_ascii=False)
            + "\n\nTarget: " + question
         )}
    ]

    client = Client(host=host) if host else None
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": False,                      # easier to parse
        "options": {"temperature": temperature},
    }

    # Use JSON Schema (structured outputs) for guaranteed shape
    if use_schema:
        kwargs["format"] = STRUCTURED_SCHEMA   # documented by Ollama blog/docs
    else:
        kwargs["format"] = "json"              # fallback: JSON syntax only

    resp = client.chat(**kwargs) if client else chat(**kwargs)
    content = resp["message"]["content"]

    # Parse JSON (structured outputs should already be valid)
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}\s*$", content, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Model did not return valid JSON:\n" + content[:600])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_json", required=True, help="Path to scene graph JSON")
    ap.add_argument("--question", required=True, help='e.g., "find a chair next to a desk"')
    ap.add_argument("--out", required=True, help="Path to save the resulting JSON (e.g., results/solutions.json)")
    ap.add_argument("--model", default="llama3.1:8b", help="Local Ollama model tag")
    ap.add_argument("--host", default=None, help="Ollama host, e.g., http://localhost:11434")
    ap.add_argument("--max_objects", type=int, default=160, help="Limit objects sent to the model")
    ap.add_argument("--no_schema", action="store_true", help="Disable JSON-schema structured outputs")
    args = ap.parse_args()

    scene = load_scene(args.scene_json)
    result = call_ollama(
        scene=scene,
        question=args.question,
        model=args.model,
        host=args.host,
        max_objects=args.max_objects,
        use_schema=not args.no_schema
    )

    # ---- Save to file (do not print JSON to stdout) ----
    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    # Optional: print a short confirmation (not the JSON payload)
    print(f"Saved solutions to {out_path}")


if __name__ == "__main__":
    main()

