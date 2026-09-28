import json
from dataclasses import dataclass

import numpy as np

from .cases import Case
from .compare import Deviation, assert_close, tolerance
from .results import Atoms, EVBRecord, Result, read_forces, write_forces


@dataclass
class Golden:
    pe: float
    f_evb: float
    evb: EVBRecord
    atoms: Atoms
    meta: dict


def load_golden(case: Case) -> Golden:
    data = json.loads(case.golden_json.read_text())
    return Golden(
        pe=data["pe"],
        f_evb=data["f_evb"],
        evb=EVBRecord.from_json(data["evb"]),
        atoms=read_forces(case.golden_forces),
        meta=data["meta"],
    )


def save_golden(case: Case, result: Result, meta: dict) -> None:
    data = {
        "case": case.id,
        "pe": result.pe[0],
        "f_evb": result.f_evb[0],
        "evb": result.evb.to_json(),
        "meta": meta,
    }
    case.golden_json.write_text(json.dumps(data, indent=2) + "\n")
    header = f"golden forces for {case.id} (partition 0, layout {result.layout})"
    write_forces(case.golden_forces, result.atoms[0], header)


def check_structure(result: Result, golden: Golden) -> None:
    if result.evb.keys != golden.evb.keys:
        missing = set(golden.evb.keys) - set(result.evb.keys)
        extra = set(result.evb.keys) - set(golden.evb.keys)
        raise AssertionError(
            f"EVB state set differs from golden ({result.evb.nstates} vs {golden.evb.nstates} states); "
            f"missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}"
        )


def check_topology(atoms: Atoms, golden: Golden) -> None:
    if not np.array_equal(atoms.ids, golden.atoms.ids):
        raise AssertionError("atom ids differ from golden")
    for name in ("types", "charges"):
        got, want = getattr(atoms, name), getattr(golden.atoms, name)
        bad = np.flatnonzero(got != want)
        if bad.size:
            i = bad[0]
            raise AssertionError(
                f"{bad.size} atom {name} differ from golden (a permanent transfer went differently); "
                f"first at id {atoms.ids[i]}: got {got[i]}, expected {want[i]}"
            )


def compare_energy(case: Case, result: Result, golden: Golden) -> dict[str, Deviation]:
    tol = tolerance(case, "energy")
    return {
        "pe": assert_close("pe", result.pe[0], golden.pe, **tol),
        "f_evb": assert_close("f_evb", result.f_evb[0], golden.f_evb, **tol),
        "eigenvalues": assert_close("eigenvalues", result.evb.eigenvalues, golden.evb.eigenvalues, **tol),
    }


def compare_evb(case: Case, result: Result, golden: Golden) -> dict[str, Deviation]:
    check_structure(result, golden)
    keys = golden.evb.keys
    out = {
        "hamiltonian": assert_close(
            "hamiltonian",
            result.evb.hamiltonian,
            golden.evb.hamiltonian,
            **tolerance(case, "hamiltonian"),
            index_labels=lambda ij: (keys[ij[0]], keys[ij[1]]),
        ),
        "amplitudes": assert_close(
            "amplitudes",
            result.evb.amplitudes,
            golden.evb.amplitudes,
            **tolerance(case, "amplitudes"),
            index_labels=lambda i: keys[i[0]],
        ),
    }
    if golden.evb.occupations is not None:
        out["occupations"] = assert_close(
            "occupations", result.evb.occupations, golden.evb.occupations, **tolerance(case, "amplitudes")
        )
    return out


def compare_forces(case: Case, atoms: Atoms, golden: Golden, label: str = "forces") -> Deviation:
    check_topology(atoms, golden)
    ids = golden.atoms.ids
    return assert_close(
        label,
        atoms.forces,
        golden.atoms.forces,
        **tolerance(case, "forces"),
        index_labels=lambda ij: f"atom id {ids[ij[0]]}, component {'xyz'[ij[1]]}",
    )
