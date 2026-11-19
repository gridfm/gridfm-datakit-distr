"""Reference doc of the generation config, generated from the models.

    uv run python -m gridfm_datakit_distr.config.reference > docs/manual/config.md

A test keeps ``docs/manual/config.md`` equal to :func:`reference_markdown`.
"""

from __future__ import annotations

import inspect
import json
import types
from typing import Any, Literal, Union, get_args, get_origin

import annotated_types as at
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from gridfm_datakit_distr.config.schema import DocType, GenerationConfig, SpecRange
from gridfm_datakit_distr.config.value_spec import ValueSpec

_INTRO = """\
# Configuration reference

<!-- Generated from `config/schema.py` by
     `uv run python -m gridfm_datakit_distr.config.reference > docs/manual/config.md`;
     do not edit. -->

A dataset is described by one YAML file (examples in `scripts/config/`), validated by
`gridfm_datakit_distr.config.load_config`. Unknown keys, non-finite numbers and values of the
wrong type are errors (`"1"` is not an integer; YAML reads `1e-10` as text, write `1.0e-10`);
every problem is reported as `<dotted.path>: <reason>`. Defaults leave the source networks
unchanged (no perturbation, no DER).

**Value specs** (gridfm-datakit's semantics): a number, or
`{distribution: normal, mean, std, min?, max?}` (truncated: out-of-range draws are redrawn, an
unbounded side ends 6 std beyond the mean), `{distribution: uniform, low, high}` (on
`[low, high)`), `{distribution: choice, values, weights?}`. A field's range applies to every
value a spec can draw: a normal needs an explicit `max` under a ceiling.
"""


def reference_markdown() -> str:
    """The reference doc: one table per section of :class:`GenerationConfig`."""
    out = [_INTRO]
    _section(GenerationConfig, "", out)
    return "\n".join(out).rstrip() + "\n"


def _section(model: type[BaseModel], path: str, out: list[str]) -> None:
    rows: list[str] = []
    children: list[tuple[str, type[BaseModel]]] = []
    for name, field in model.model_fields.items():
        dotted = f"{path}.{name}" if path else name
        sub = _model_of(field.annotation)
        if sub is not None:
            children.append((dotted, sub))
            continue
        rows.append(
            f"| `{dotted}` | {_type(field)} | {_default(field)} | {field.description or ''} |"
        )
    title = f"## `{path}`" if path else "## Top level"
    out += [title, "", inspect.cleandoc(model.__doc__ or ""), ""]
    if rows:
        out += ["| Key | Type | Default | Description |", "|---|---|---|---|", *rows, ""]
    for dotted, sub in children:
        _section(sub, dotted, out)


def _model_of(annotation: Any) -> type[BaseModel] | None:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _type(field: FieldInfo) -> str:
    nullable = _nullable(field.annotation)
    metadata = list(field.metadata)
    if nullable:  # an Annotated member of a union keeps its metadata
        for arg in get_args(field.annotation):
            metadata += _unwrap(arg)[1]
    for meta in metadata:
        if isinstance(meta, DocType):
            return meta.text
        if isinstance(meta, SpecRange):
            return f"{meta.describe()} or null" if nullable else meta.describe()
    return _describe(field.annotation, metadata)


def _nullable(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType) and type(None) in get_args(annotation)


def _unwrap(annotation: Any) -> tuple[Any, list[Any]]:
    """The type inside ``Annotated`` and its metadata (constraints of ``Field`` included)."""
    if not hasattr(annotation, "__metadata__"):
        return annotation, []
    metadata: list[Any] = []
    for meta in annotation.__metadata__:
        metadata += meta.metadata if isinstance(meta, FieldInfo) else [meta]
    return get_args(annotation)[0], metadata


def _describe(annotation: Any, metadata: list[Any]) -> str:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        parts = []
        for arg in args:
            base, inner = _unwrap(arg)
            parts.append(_describe(base, metadata + inner))
        text = " or ".join(parts)
        return f"{text} or null" if len(args) < len(get_args(annotation)) else text
    if origin is Literal:
        return "one of " + ", ".join(f"`{v}`" for v in get_args(annotation))
    if origin is list:
        (item,) = get_args(annotation)
        base, inner = _unwrap(item)
        if get_origin(base) is Literal:
            text = "list of " + ", ".join(f"`{v}`" for v in get_args(base))
        else:
            text = "list of " + _plural(_describe(base, []))
        each = _bounds(inner).strip()
        outer = _bounds(metadata, length=True).strip()
        return text + (f" (each {each})" if each else "") + (f", {outer}" if outer else "")
    names = {bool: "boolean", int: "integer", float: "number", str: "string"}
    return names.get(annotation, getattr(annotation, "__name__", str(annotation))) + _bounds(
        metadata
    )


def _plural(text: str) -> str:
    return {"string": "strings", "integer": "integers", "number": "numbers"}.get(text, text)


def _bounds(metadata: list[Any], *, length: bool = False) -> str:
    parts = []
    for meta in metadata:
        if isinstance(meta, at.Gt):
            parts.append(f"> {_number(meta.gt)}")
        elif isinstance(meta, at.Ge):
            parts.append(f"≥ {_number(meta.ge)}")
        elif isinstance(meta, at.Lt):
            parts.append(f"< {_number(meta.lt)}")
        elif isinstance(meta, at.Le):
            parts.append(f"≤ {_number(meta.le)}")
        elif isinstance(meta, at.MinLen):
            n = meta.min_length
            unit = "item" if length else "character"
            parts.append(f"{'at least' if length else 'of at least'} {n} {unit}{'s' * (n != 1)}")
        elif getattr(meta, "pattern", None):
            parts.append(f"matching `{meta.pattern}`")
    return " " + " and ".join(parts) if parts else ""


def _number(value: Any) -> str:
    return str(value) if isinstance(value, int) else f"{value:g}"


def _default(field: FieldInfo) -> str:
    if field.default is PydanticUndefined and field.default_factory is None:
        return "required"
    value = field.get_default(call_default_factory=True, validated_data={})
    if isinstance(value, ValueSpec):
        value = value.to_config()
    if value is None:
        return "null"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {_scalar(v)}" for k, v in value.items())
        return f"`{{{inner}}}`"
    return f"`{_scalar(value)}`"


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    if isinstance(value, str):
        return value
    return json.dumps(value)


if __name__ == "__main__":
    print(reference_markdown(), end="")
