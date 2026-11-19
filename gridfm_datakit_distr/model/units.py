"""Constants of the three-phase conventions (SI internally; per unit only in exported views).

With ``v_ll`` the line-to-line voltage, ``v_ln = v_ll / sqrt(3)``. A split-phase zone (0.12 kV
halves) has ``v_ln = 0.12 kV`` and the three-phase-equivalent ``v_ll = 0.208 kV``, as ``Bus.kVBase``
in OpenDSS.
"""

from __future__ import annotations

import math

SQRT3 = math.sqrt(3.0)
