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

A generation config is one YAML file that describes one dataset. This page lists every key with
its type, range, default and effect, generated from the models that validate it.

## Overview

The sections follow the order in which a dataset is built:

1. `networks`, `normalisation`: select catalog networks, normalise their models
   ([Networks](networks.md)).
2. `variants`, `der`: perturb each network into variants and place PV ([Variants](variants.md)).
3. `scenarios`: on each variant, draw the loads, PV output and source voltage of each scenario
   ([Scenarios](scenarios.md)).
4. `controls`: settle each scenario's regulators and capacitors into a network state.
5. `labels`, `views`: solve the power flow at that state; optional element currents, OPF
   ([OPF labels](opf.md)) and Kron view.
6. `settings`: run in chunks and write ([Outputs](outputs.md)).

Four keys are required; every other key has a default:

```yaml
dataset_name: my_dataset
networks: {include: [ieee13]}
scenarios: {n_scenarios_per_variant: 100}
settings: {seed: 1}
```

Sample configs in `scripts/config/`:

| Config | Shows |
|---|---|
| `default.yaml` | a commented starting point; every optional key it sets is at its default |
| `ieee13_small.yaml` | the smallest power-flow dataset: IEEE 13, 2 variants of 20 scenarios |
| `ieee123.yaml` | regulators and capacitors settled per scenario; ANSI voltage limits |
| `csiro_feeder.yaml` | a four-wire LV feeder: phase reallocation, PV |
| `csiro_enwl_pf.yaml` | most keys written out; Kron view on |
| `edp_rural.yaml` | loads from EDP's smart-meter profiles; EN 50160 voltage limits |
| `opf_small.yaml` | OPF labels on one LV feeder in seconds |
| `opf_slice.yaml` | OPF labels on five LV feeders with PV that pushes voltages over the limit |
| `sample.yaml` | the v0.2 sample dataset: IEEE 13 and EDP rural, power-flow and OPF labels |
| `health.yaml` | a small catalog health run: every network but `csiro_enwl/n13_f4` (fails C1) |

## Checking a config

`gridfm_datakit_distr generate <config> --dry-run` validates the config, loads its networks (they
must be fetched) and prints the counts of scenarios and work units with time and storage
estimates; it writes nothing. An invalid config stops it with exit code 2 and every problem listed
at once, one line each: `<dotted.path>: <reason>` (list items as `[i]`). In Python,
`gridfm_datakit_distr.config.load_config(path)` runs the same config checks without the network
cache.

- Unknown keys are errors; a missing key takes its default.
- Types are not coerced: `"1"` is not a number, `true` is not an integer (an integer is a valid
  number); `.inf` and `.nan` are refused.
- YAML 1.1: an exponent needs a decimal point (`1.0e-10`; `1e-10` is read as text); a
  duplicated key is an error.
- Units are SI unless the key's name says otherwise (`_pu`, `kwp` in kW).

**Dataset identity.** The dataset's `manifest.json` stores the resolved config, every default
filled in; its `config_hash` covers every key except `settings.data_dir`, `num_processes`,
`contexts_per_process` and `overwrite`. When `generate` resumes, refuses or overwrites a dataset:
[Command-line reference](cli.md#generate).

## Value specs

A key of type *value spec* takes a number or a distribution, drawn anew each time its description
says (per variant, per scenario, per unit):

- `0.5`: always 0.5.
- `{distribution: uniform, low: 0.9, high: 1.1}`: uniform on [low, high); low < high.
- `{distribution: normal, mean: 1.0, std: 0.05, min: 0.9, max: 1.1}`: normal truncated to
  [min, max], each bound optional; std > 0; out-of-range draws are redrawn, never clipped.
- `{distribution: choice, values: [1, 5, 20], weights: [2, 1, 1]}`: one of `values`, with
  probabilities proportional to `weights` (non-negative, positive sum; default: equal).

The range in the Type column holds for every value a spec can draw. Under a lower bound, a fixed
number, a uniform's `low` or any choice below it is an error, while a normal is truncated at it.
Under an upper bound, the largest possible draw must respect it: a uniform's `high`, the largest
choice, the end of a normal. A normal's unbounded side ends 6 std beyond its mean (or beyond its
other bound, when that is further out), so under an upper bound it needs a `max` unless
mean + 6 std fits; a `[min, max]` outside mean ± 6 std is an error.
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
    level = "#" * (2 + path.count("."))  # nested sections nest in the page's table of contents
    title = f"{level} `{path}`" if path else "## Top level"
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
