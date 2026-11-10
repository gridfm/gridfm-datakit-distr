"""Taxonomy of OpenDSS element classes in the electrical model.

Single source of truth for the extractor and the coverage report. Class names are lower case.
"""

from __future__ import annotations

from typing import Literal

Category = Literal["pd", "pc", "source", "control", "unsupported"]

PD_CLASSES = frozenset({"line", "transformer", "capacitor", "reactor"})
"""Power delivery: their Yprims form the network admittance."""
PC_CLASSES = frozenset({"load", "generator", "pvsystem", "storage"})
"""Power conversion: injections, not part of the network admittance."""
SOURCE_CLASSES = frozenset({"vsource"})
"""The slack."""
CONTROL_CLASSES = frozenset(
    {"regcontrol", "capcontrol", "swtcontrol", "fuse", "recloser", "relay", "energymeter",
     "monitor", "sensor", "storagecontroller", "gendispatcher"}
)  # fmt: skip
"""Controllers, protection and meters: skipped by the electrical layer. Inverter controls
(InvControl, ExpControl) change PV behaviour and are unsupported in v1."""


def category(element_class: str) -> Category:
    """The category of a lower-case OpenDSS class name; anything unknown is unsupported."""
    for name, classes in (("pd", PD_CLASSES), ("pc", PC_CLASSES), ("source", SOURCE_CLASSES),
                          ("control", CONTROL_CLASSES)):  # fmt: skip
        if element_class in classes:
            return name  # type: ignore[return-value]
    return "unsupported"
