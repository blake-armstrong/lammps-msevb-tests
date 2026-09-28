from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cases import Case
from .results import Result
from .runner import Layout, run_case


@dataclass
class FDForce:
    atom_id: int
    force: np.ndarray
    step: float
    states_stable: bool


def displacement_commands(atom_id: int, vector) -> str:
    dx, dy, dz = (f"{v:.17g}" for v in vector)
    return f"group fd_atom id {atom_id}\ndisplace_atoms fd_atom move {dx} {dy} {dz} units box\n"


def displaced_run(
    case: Case, layout: Layout, workdir: Path, atom_id: int, vector, extra_commands: str = ""
) -> Result:
    commands = extra_commands + "\n" + displacement_commands(atom_id, vector)
    return run_case(case, layout, workdir, extra_commands=commands, mode="run0")


def reactive_atoms(result: Result) -> list[int]:
    ids = set()
    for _, links in result.evb.keys:
        for x, h, y, _ in links:
            ids.update(i for i in (x, h, y) if i > 0)
    return sorted(ids)


def fd_force(
    case: Case,
    layout: Layout,
    workdir: Path,
    atom_id: int,
    reference: Result,
    step: float = 1.0e-4,
    extra_commands: str = "",
) -> FDForce:
    force = np.zeros(3)
    stable = True
    for c in range(3):
        vector = np.zeros(3)
        vector[c] = step
        base = Path(workdir) / f"{atom_id}-{step:g}-{'xyz'[c]}"
        plus = displaced_run(case, layout, Path(f"{base}+"), atom_id, vector, extra_commands)
        minus = displaced_run(case, layout, Path(f"{base}-"), atom_id, -vector, extra_commands)
        force[c] = -(plus.pe[0] - minus.pe[0]) / (2.0 * step)
        for r in (plus, minus):
            stable &= r.evb.keys == reference.evb.keys
            stable &= bool(np.array_equal(r.atoms[0].types, reference.atoms[0].types))
    return FDForce(atom_id=atom_id, force=force, step=step, states_stable=stable)
