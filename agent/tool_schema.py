"""Token-efficient tool declarations for the Laby agent.

Every tool schema is re-sent on every LLM call, so the fixed prefix (system
prompt + 48 tool declarations) dominates our per-turn cost. ADK builds those
declarations from Pydantic, which emits three kinds of pure overhead:

  1. ``title`` on every property ("Is E Signed") — a Pydantic artifact that
     tells the model nothing it cannot read from the property name.
  2. ``anyOf: [{type: X}, {type: "null"}]`` for every Optional[...] param —
     semantically identical to ``type: X`` with a null default.
  3. The Google-style ``Args:`` block, which ADK leaves in the *description*
     prose while separately emitting a parameter schema. The argument docs
     therefore ship twice: once as indented prose, once as JSON.

This module removes 1 and 2 (both are lossless by construction) and relocates
3 — each argument's documentation is moved onto its own schema property, so
the model still receives every word, just once and in the place the
function-calling API expects it.

No wording is discarded. If an ``Args:`` entry has no matching property the
prose is left untouched, so a docstring typo degrades to the old behaviour
rather than silently dropping guidance.
"""

import re
from typing import Any, Dict, Optional, Tuple

from google.adk.tools.function_tool import FunctionTool

# "    name: text" at exactly one indent level inside the Args: block.
_ARG_LINE = re.compile(r"^\s{4}(\w+):\s*(.*)$")


def _split_args_block(description: str) -> Tuple[str, Dict[str, str]]:
    """Split a Google-style docstring into (prose, {arg_name: arg_doc})."""
    idx = description.find("\nArgs:")
    if idx == -1:
        return description, {}

    prose = description[:idx].rstrip()
    body = description[idx + len("\nArgs:"):]

    args: Dict[str, str] = {}
    current: Optional[str] = None
    for line in body.split("\n"):
        if not line.strip():
            continue
        m = _ARG_LINE.match(line)
        if m:
            current = m.group(1)
            args[current] = m.group(2).strip()
        elif current and line.startswith(" " * 5):
            # Continuation of the previous arg's description.
            args[current] = f"{args[current]} {line.strip()}".strip()
        else:
            # Dedented back out of Args: (e.g. a Returns: block) — stop.
            current = None
    return prose, args


def _strip_pydantic_noise(node: Any) -> Any:
    """Drop ``title`` and collapse ``anyOf: [T, null]`` to a bare type."""
    if isinstance(node, dict):
        node.pop("title", None)
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            non_null = [a for a in any_of
                        if isinstance(a, dict) and a.get("type") != "null"]
            if len(non_null) == 1:
                node.pop("anyOf")
                # Preserve any sibling keys already on the node (default, ...).
                for k, v in non_null[0].items():
                    node.setdefault(k, v)
        for value in node.values():
            _strip_pydantic_noise(value)
    elif isinstance(node, list):
        for value in node:
            _strip_pydantic_noise(value)
    return node


class SlimFunctionTool(FunctionTool):
    """FunctionTool whose declaration carries the same information, fewer tokens."""

    def _get_declaration(self):  # type: ignore[override]
        decl = super()._get_declaration()
        if decl is None:
            return decl

        schema = decl.parameters_json_schema
        if not isinstance(schema, dict):
            return decl

        prose, arg_docs = _split_args_block(decl.description or "")
        props = schema.get("properties")

        if arg_docs and isinstance(props, dict):
            unmatched = False
            for name, doc in arg_docs.items():
                target = props.get(name)
                if isinstance(target, dict):
                    # Relocate the prose onto the property itself.
                    target.setdefault("description", doc)
                else:
                    unmatched = True
            # Only drop the prose block once every arg found a home.
            if not unmatched:
                decl.description = prose

        decl.parameters_json_schema = _strip_pydantic_noise(schema)
        return decl


def slim_tools(functions) -> list:
    """Wrap plain tool functions as token-efficient ADK FunctionTools."""
    return [SlimFunctionTool(func=fn) for fn in functions]
