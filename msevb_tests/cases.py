import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASES_DIR = REPO / "cases"
COMMON_DIR = CASES_DIR / "common"
GROUPS = ("examples", "bicarbonate")

DEFAULT_MIN_COUPLING = {"metal": 0.01, "real": 0.23}
MODES = ("run0", "minimize")
CRITERIA = ("coupling", "fermi_dirac")
NOT_RUN_INPUTS = {"case.toml", "golden.json", "golden_forces.dat", "in.setup"}
SOURCE_ROOTS = {
    "lammps-fork": ("LAMMPS_FORK", Path.home() / "git" / "lammps-fork"),
    "bicarbonate-msevb": ("BICARBONATE_MSEVB", Path.home() / "git" / "bicarbonate-msevb"),
}


@dataclass(frozen=True)
class Case:
    group: str
    name: str
    path: Path
    source: str
    units: str
    mode: str
    criterion: str
    min_coupling: float
    min_lowering: float
    min_weight: float
    files: tuple[str, ...] = ()
    tolerances: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def setup_file(self) -> Path:
        return self.path / "in.setup"

    @property
    def golden_json(self) -> Path:
        return self.path / "golden.json"

    @property
    def golden_forces(self) -> Path:
        return self.path / "golden_forces.dat"

    @property
    def source_dir(self) -> Path:
        root, _, rel = self.source.partition(":")
        env, default = SOURCE_ROOTS[root]
        return Path(os.environ.get(env, default)) / rel

    def input_files(self) -> list[Path]:
        return sorted(p for p in self.path.iterdir() if p.is_file() and p.name not in NOT_RUN_INPUTS)


def load_case(path: Path) -> Case:
    path = Path(path).resolve()
    spec = tomllib.loads((path / "case.toml").read_text())
    units = spec["units"]
    mode = spec.get("mode", "run0")
    criterion = spec.get("criterion", "coupling")
    if mode not in MODES:
        raise ValueError(f"{path}: mode must be one of {MODES}, got {mode!r}")
    if criterion not in CRITERIA:
        raise ValueError(f"{path}: criterion must be one of {CRITERIA}, got {criterion!r}")
    return Case(
        group=path.parent.name,
        name=path.name,
        path=path,
        source=spec["source"],
        units=units,
        mode=mode,
        criterion=criterion,
        min_coupling=spec.get("min_coupling", DEFAULT_MIN_COUPLING[units]),
        min_lowering=spec.get("min_lowering", DEFAULT_MIN_COUPLING[units]),
        min_weight=spec.get("min_weight", 1.0e-3),
        files=tuple(spec.get("files", ())),
        tolerances=spec.get("tolerances", {}),
    )


def discover(group: str | None = None) -> list[Case]:
    groups = GROUPS if group is None else (group,)
    cases = []
    for g in groups:
        for toml in sorted((CASES_DIR / g).glob("*/case.toml")):
            cases.append(load_case(toml.parent))
    return cases


def find_case(case_id: str) -> Case:
    for case in discover():
        if case_id in (case.id, case.name):
            return case
    raise KeyError(f"no case named {case_id!r}")
