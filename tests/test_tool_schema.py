"""The slim tool declarations must stay lossless.

These guard the prompt-cost optimisation: the declarations may get smaller,
but no argument documentation and no parameter may ever disappear.
"""

import json

import pytest
from google.adk.tools.function_tool import FunctionTool

from agent.tool_schema import SlimFunctionTool, _split_args_block, slim_tools
from agent.tools import LABY_TOOLS


def _decl(tool_cls, fn):
    return tool_cls(func=fn)._get_declaration()


def _props(decl):
    schema = decl.parameters_json_schema or {}
    return set((schema.get("properties") or {}).keys())


@pytest.mark.parametrize("fn", LABY_TOOLS, ids=lambda f: f.__name__)
def test_same_parameters_as_default_tool(fn):
    """Slimming must never add or drop a parameter."""
    assert _props(_decl(SlimFunctionTool, fn)) == _props(_decl(FunctionTool, fn))


@pytest.mark.parametrize("fn", LABY_TOOLS, ids=lambda f: f.__name__)
def test_arg_documentation_is_preserved(fn):
    """Every documented arg must still be reachable by the model."""
    base = _decl(FunctionTool, fn)
    slim = _decl(SlimFunctionTool, fn)
    _, arg_docs = _split_args_block(base.description or "")
    reachable = (slim.description or "") + json.dumps(slim.parameters_json_schema or {})
    for name, doc in arg_docs.items():
        assert name in reachable, f"{fn.__name__}: arg {name} vanished"
        for word in [w for w in doc.split() if len(w) > 5][:4]:
            assert word.strip('",.()') in reachable, (
                f"{fn.__name__}.{name}: lost documentation {word!r}")


def test_no_pydantic_noise_remains():
    blob = json.dumps([
        (d.description, d.parameters_json_schema)
        for d in (_decl(SlimFunctionTool, fn) for fn in LABY_TOOLS)
    ])
    assert '"title"' not in blob
    assert '"anyOf"' not in blob


def test_declarations_are_actually_smaller():
    def size(cls):
        return len(json.dumps([
            (d.name, d.description, d.parameters_json_schema)
            for d in (_decl(cls, fn) for fn in LABY_TOOLS)
        ]))

    assert size(SlimFunctionTool) < size(FunctionTool) * 0.95


def test_slim_tools_wraps_every_tool():
    tools = slim_tools(LABY_TOOLS)
    assert len(tools) == len(LABY_TOOLS)
    assert all(isinstance(t, SlimFunctionTool) for t in tools)
