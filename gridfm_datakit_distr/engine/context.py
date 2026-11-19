"""OpenDSS engine context: one DSS-Extensions engine per instance, one instance per thread.

The engine facts this module relies on (array layouts, error reporting, working-directory
behaviour, option codes) were confirmed by experiment before the code relied on them.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from opendssdirect import DSSException
from opendssdirect.OpenDSSDirect import OpenDSSDirect

from gridfm_datakit_distr.engine.errors import DssError

ControlMode = Literal["OFF", "STATIC"]


class SolveStatus(NamedTuple):
    """Outcome of one power-flow solve."""

    converged: bool
    iterations: int


def complex_from_interleaved(values: ArrayLike) -> NDArray[np.complex128]:
    """Convert an OpenDSS legacy array ``[re0, im0, re1, im1, ...]`` to complex128.

    This is the only place where the interleaved layout is decoded.
    """
    arr = np.ascontiguousarray(values)
    if arr.dtype != np.float64 or arr.ndim != 1 or arr.size % 2:
        raise ValueError(
            "expected a flat float64 array of even length (re/im interleaved), "
            f"got dtype={arr.dtype}, shape={arr.shape}"
        )
    return arr.view(np.complex128)


def matrix_from_yprim(values: ArrayLike) -> NDArray[np.complex128]:
    """Convert a legacy ``CktElement.YPrim()`` array to an ``(n, n)`` complex matrix.

    OpenDSS returns the matrix column by column (column-major), as interleaved re/im pairs.
    """
    flat = complex_from_interleaved(values)
    n = math.isqrt(flat.size)
    if n * n != flat.size:
        raise ValueError(f"YPrim has {flat.size} complex entries, not a square number")
    return flat.reshape((n, n), order="F")


_FACTORY_LOCK = threading.Lock()
_factory: Any = None


def _new_engine() -> Any:
    """Create an independent engine whose array getters return NumPy arrays.

    ``NewContext()`` inherits the array preference of the instance it is called on, and the
    default instance returns Python lists unless ``OPENDSSDIRECT_PY_USE_NUMPY`` was set before
    import. A NumPy-preferring instance on the prime engine is therefore used as factory.
    """
    global _factory
    with _FACTORY_LOCK:
        if _factory is None:
            _factory = OpenDSSDirect(prefer_lists=False)
        return _factory.NewContext()


def _unpack(exc: DSSException) -> tuple[int | None, str]:
    """Split a DSS-Python exception into (error number, message)."""
    if len(exc.args) == 2 and isinstance(exc.args[0], int):
        return exc.args[0], str(exc.args[1]).strip()
    return None, str(exc).strip()


class DssContext:
    """One OpenDSS engine context.

    Not thread-safe: create one per thread and never share it. Solver options are
    reset by ``Clear``, so call :meth:`set_options` after :meth:`load`.
    """

    def __init__(self, engine: Any) -> None:
        """Wrap an OpenDSSDirect.py engine; use :meth:`new` to create one."""
        self._dss = engine
        self.load_actions: list[str] = []
        """Commands that :meth:`load` added on its own initiative (for the passport)."""

    def __del__(self) -> None:
        # DSS-Python (0.15) never disposes a context: without this, each dropped one keeps its
        # compiled circuit (tens of MB on large feeders) until the process exits.
        with contextlib.suppress(Exception):  # nothing to report to while being collected
            self._dss.Text.Command("Clear")

    @classmethod
    def new(cls) -> DssContext:
        """Create a new engine context that never changes the process working directory."""
        engine = _new_engine()
        engine.Basic.AllowChangeDir(False)
        if engine.Basic.AdvancedTypes():
            raise DssError(
                "NewContext",
                "AdvancedTypes is enabled; this wrapper decodes the legacy flat array layout",
            )
        return cls(engine)

    @property
    def api(self) -> Any:
        """The underlying OpenDSSDirect.py instance (for read-only class interfaces)."""
        return self._dss

    @contextlib.contextmanager
    def engine_call(self, what: str) -> Iterator[None]:
        """Translate DSS-Python exceptions raised inside the block into :class:`DssError`."""
        try:
            yield
        except DSSException as exc:
            code, message = _unpack(exc)
            raise DssError(what, message, code) from exc

    def command(self, cmd: str) -> str:
        """Run one OpenDSS command and return its result text (``Text.Result``)."""
        with self.engine_call(cmd):
            self._dss.Text.Command(cmd)
            return str(self._dss.Text.Result())

    def load(self, master: Path, extra_commands: Sequence[str] = ()) -> None:
        """Compile a circuit from its master file.

        Runs ``Clear``, ``Redirect "<absolute master path>"`` and ``extra_commands``; then, if
        some bus has no voltage base, ``CalcVoltageBases`` (recorded in :attr:`load_actions`).
        Relative includes resolve against the including file's folder; the process working
        directory is never changed.

        ``CalcVoltageBases`` assigns each bus the nearest base of the ``VoltageBases`` list
        (OpenDSS defaults if the model sets none), however far it is: the bases themselves
        are validated by the normalisation step, not here.
        """
        path = Path(master).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if '"' in str(path):
            raise ValueError(f"the OpenDSS parser cannot read a path containing '\"': {path}")
        self.load_actions = []
        self.command("Clear")
        self.command(f'Redirect "{path}"')
        for cmd in extra_commands:
            self.command(cmd)
        if self.buses_without_voltage_base():
            self.command("CalcVoltageBases")
            self.load_actions.append("CalcVoltageBases")

    def buses_without_voltage_base(self) -> list[str]:
        """Names (lower case) of the buses whose ``kVBase`` is not positive.

        The bus list does not exist right after compilation (it is built by
        ``CalcVoltageBases`` or the first solve), so it is built here with ``MakeBusList``.
        """
        missing: list[str] = []
        self.command("MakeBusList")
        with self.engine_call("Bus.kVBase"):
            for name in self._dss.Circuit.AllBusNames():
                self._dss.Circuit.SetActiveBus(name)
                if not self._dss.Bus.kVBase() > 0:
                    missing.append(name.lower())
        return missing

    def set_options(
        self,
        *,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
        control_mode: ControlMode = "OFF",
    ) -> None:
        """Set the solver tolerance (pu voltage change), iteration cap and control mode."""
        if not tolerance > 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if control_mode not in ("OFF", "STATIC"):
            raise ValueError(f"control_mode must be OFF or STATIC, got {control_mode!r}")
        with self.engine_call("set_options"):
            self._dss.Solution.Convergence(tolerance)
            self._dss.Solution.MaxIterations(max_iterations)
        self.command(f"Set ControlMode={control_mode}")

    def solve(self) -> SolveStatus:
        """Solve the power flow. Non-convergence is reported in the status, not raised."""
        with self.engine_call("Solve"):
            self._dss.Solution.Solve()
            return SolveStatus(
                converged=bool(self._dss.Solution.Converged()),
                iterations=int(self._dss.Solution.Iterations()),
            )

    def ynode_order(self) -> list[str]:
        """Node names ``bus.conductor`` (lower case) in the engine's Y-matrix order."""
        with self.engine_call("Circuit.YNodeOrder"):
            return [name.lower() for name in self._dss.Circuit.YNodeOrder()]

    def ynode_positions(self, names: Iterable[str]) -> NDArray[np.intp]:
        """Position of each node name (``bus.conductor``, any case) in :meth:`ynode_order`; a
        name the engine does not have raises ``KeyError``."""
        order = {name: j for j, name in enumerate(self.ynode_order())}
        return np.array([order[name.lower()] for name in names], dtype=np.intp)

    def ynode_voltages(self) -> NDArray[np.complex128]:
        """Node voltages to ground (V), ordered as :meth:`ynode_order`."""
        with self.engine_call("Circuit.YNodeVArray"):
            return complex_from_interleaved(self._dss.Circuit.YNodeVArray())

    def activate_element(self, full_name: str) -> None:
        """Make ``full_name`` (e.g. ``Line.l1``) the active circuit element."""
        what = f"SetActiveElement {full_name}"
        with self.engine_call(what):
            index = self._dss.Circuit.SetActiveElement(full_name)
        if index < 0:
            # The engine keeps the previous element active: never read on after this.
            raise DssError(what, "element not found")

    def element_yprim(self, full_name: str) -> NDArray[np.complex128]:
        """Primitive admittance matrix (S) of an element, ordered as its ``NodeOrder``."""
        self.activate_element(full_name)
        with self.engine_call(f"CktElement.YPrim {full_name}"):
            return matrix_from_yprim(self._dss.CktElement.YPrim())

    def system_y(self) -> sp.csr_matrix:
        """The system admittance matrix (S) the solver used in the last solve, ``YNodeOrder``.

        Read with ``YMatrix.getYsparse(factor=False)``; it includes PC elements and sources
        (confirmed by experiment). Never call ``BuildYMatrix`` on a labelling context.
        """
        with self.engine_call("YMatrix.getYsparse"):
            raw = self._dss.YMatrix.getYsparse(factor=False)
        if raw is None:
            raise DssError("YMatrix.getYsparse", "no system Y available: solve first")
        data, indices, indptr = raw
        n = len(indptr) - 1
        return sp.csr_matrix(sp.csc_matrix((data, indices, indptr), shape=(n, n)))

    def engine_version(self) -> str:
        """Engine and binding versions on one line (DSS C-API, DSS-Python, OpenDSSDirect.py)."""
        with self.engine_call("Basic.Version"):
            text = str(self._dss.Basic.Version())
        return " | ".join(line.strip() for line in text.splitlines() if line.strip())
