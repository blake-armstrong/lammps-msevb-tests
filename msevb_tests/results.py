import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DUMP_COLUMNS = "id type q fx fy fz"
REFERENCE_KEY = ("reference", ())


@dataclass
class Atoms:
    ids: np.ndarray
    types: np.ndarray
    charges: np.ndarray
    forces: np.ndarray

    @classmethod
    def from_table(cls, table: np.ndarray) -> "Atoms":
        table = np.atleast_2d(table)
        return cls(
            ids=table[:, 0].astype(np.int64),
            types=table[:, 1].astype(np.int64),
            charges=table[:, 2].copy(),
            forces=table[:, 3:6].copy(),
        )


@dataclass
class EVBRecord:
    keys: list[tuple]
    hamiltonian: np.ndarray
    eigenvalues: np.ndarray
    amplitudes: np.ndarray
    occupations: np.ndarray | None
    timestep: int
    coupling_scale: float = 1.0
    temperature: float | None = None

    @property
    def nstates(self) -> int:
        return len(self.keys)

    @property
    def ground_energy(self) -> float:
        return float(self.eigenvalues[0])

    def to_json(self) -> dict:
        return {
            "timestep": self.timestep,
            "nstates": self.nstates,
            "states": [key_to_json(k) for k in self.keys],
            "hamiltonian": self.hamiltonian.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "amplitudes": self.amplitudes.tolist(),
            "occupations": None if self.occupations is None else self.occupations.tolist(),
            "coupling_scale": self.coupling_scale,
            "temperature": self.temperature,
        }

    @classmethod
    def from_json(cls, data: dict) -> "EVBRecord":
        occ = data.get("occupations")
        return cls(
            keys=[key_from_json(k) for k in data["states"]],
            hamiltonian=np.array(data["hamiltonian"], dtype=float),
            eigenvalues=np.array(data["eigenvalues"], dtype=float),
            amplitudes=np.array(data["amplitudes"], dtype=float),
            occupations=None if occ is None else np.array(occ, dtype=float),
            timestep=data["timestep"],
            coupling_scale=data.get("coupling_scale", 1.0),
            temperature=data.get("temperature"),
        )


@dataclass
class Result:
    layout: str
    workdir: Path
    pe: list[float]
    f_evb: list[float]
    atoms: list[Atoms]
    evb: EVBRecord
    nrecords: int


def state_key(state: dict) -> tuple:
    links = tuple((l["X"], l["H"], l["Y"], l["reaction"]) for l in state["chain"])
    if state.get("kind") == "product":
        return ("product", tuple(sorted(links)))
    return ("chain", links)


def key_to_json(key: tuple) -> list:
    return [key[0], [list(link) for link in key[1]]]


def key_from_json(data: list) -> tuple:
    return (data[0], tuple(tuple(link) for link in data[1]))


def canonical_record(record: dict) -> EVBRecord:
    raw_keys = [REFERENCE_KEY] + [state_key(s) for s in record["states"]]
    n = record["nstates"]
    if len(raw_keys) != n:
        raise ValueError(f"record lists {len(raw_keys) - 1} states but nstates = {n}")
    if len(set(raw_keys)) != n:
        raise ValueError("record contains duplicate EVB states")

    order = [0] + sorted(range(1, n), key=lambda i: raw_keys[i])
    ham = np.array(record["hamiltonian"], dtype=float)[np.ix_(order, order)]
    amps = np.array(record["amplitudes"], dtype=float)[order]

    evals = np.array(record["eigenvalues"], dtype=float)
    eval_order = np.argsort(evals, kind="stable")
    occupations = None
    temperature = None
    if "fermi_dirac" in record:
        occupations = np.array(record["fermi_dirac"]["occupations"], dtype=float)[eval_order]
        temperature = record["fermi_dirac"]["temperature"]

    return EVBRecord(
        keys=[raw_keys[i] for i in order],
        hamiltonian=ham,
        eigenvalues=evals[eval_order],
        amplitudes=amps,
        occupations=occupations,
        timestep=record["timestep"],
        coupling_scale=record.get("coupling_scale", 1.0),
        temperature=temperature,
    )


def read_evb_json(path: Path) -> tuple[EVBRecord, int]:
    data = json.loads(Path(path).read_text())
    records = data["timesteps"]
    if not records:
        raise ValueError(f"{path} contains no EVB records")
    return canonical_record(records[-1]), len(records)


def read_dump(path: Path) -> Atoms:
    lines = Path(path).read_text().splitlines()
    header = f"ITEM: ATOMS {DUMP_COLUMNS}"
    start = lines.index(header) + 1
    return Atoms.from_table(np.loadtxt(lines[start:], ndmin=2))


def read_energy(path: Path) -> tuple[float, float]:
    pe, f_evb = Path(path).read_text().split()
    return float(pe), float(f_evb)


def parse_run(workdir: Path, npartitions: int, layout: str) -> Result:
    workdir = Path(workdir)
    energies = [read_energy(workdir / f"energy.{p}.txt") for p in range(npartitions)]
    atoms = [read_dump(workdir / f"forces.{p}.dump") for p in range(npartitions)]
    evb, nrecords = read_evb_json(workdir / "msevb.json")
    return Result(
        layout=layout,
        workdir=workdir,
        pe=[e[0] for e in energies],
        f_evb=[e[1] for e in energies],
        atoms=atoms,
        evb=evb,
        nrecords=nrecords,
    )


def write_forces(path: Path, atoms: Atoms, header: str = "") -> None:
    table = np.column_stack([atoms.ids, atoms.types, atoms.charges, atoms.forces])
    lines = [f"# {line}" for line in header.splitlines()]
    lines.append(f"# {DUMP_COLUMNS}")
    for row in table:
        lines.append(f"{int(row[0])} {int(row[1])} " + " ".join(f"{v:.17g}" for v in row[2:]))
    Path(path).write_text("\n".join(lines) + "\n")


def read_forces(path: Path) -> Atoms:
    return Atoms.from_table(np.loadtxt(path, comments="#", ndmin=2))
