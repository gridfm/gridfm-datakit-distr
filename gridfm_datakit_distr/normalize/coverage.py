"""Element coverage of a compiled circuit: classes, counts and support status (check C4)."""

from __future__ import annotations

from dataclasses import dataclass

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.classes import Category, category


@dataclass(frozen=True, slots=True)
class ClassCount:
    """Elements of one OpenDSS class."""

    element_class: str
    category: Category
    enabled: int
    disabled: int


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Per-class element counts of a circuit."""

    classes: tuple[ClassCount, ...]

    @property
    def unsupported_enabled(self) -> list[str]:
        """Classes with enabled elements the model does not support."""
        return [c.element_class for c in self.classes if c.category == "unsupported" and c.enabled]

    def passes(self) -> bool:
        """C4: no enabled element of an unsupported class."""
        return not self.unsupported_enabled


def coverage_report(ctx: DssContext) -> CoverageReport:
    """Count the elements of every class in the compiled circuit of ``ctx``."""
    api = ctx.api
    counts: dict[str, list[int]] = {}
    for name in api.Circuit.AllElementNames():
        ctx.activate_element(name)
        with ctx.engine_call(f"CktElement.Enabled {name}"):
            enabled = bool(api.CktElement.Enabled())
        entry = counts.setdefault(name.split(".", 1)[0].lower(), [0, 0])
        entry[0 if enabled else 1] += 1
    return CoverageReport(
        tuple(
            ClassCount(cls, category(cls), enabled, disabled)
            for cls, (enabled, disabled) in sorted(counts.items())
        )
    )
