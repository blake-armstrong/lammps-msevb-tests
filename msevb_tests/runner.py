import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .cases import COMMON_DIR, Case
from .results import Result, parse_run

FILE_KEYWORD = "file msevb.json every 1"
DEFAULT_LMP = Path.home() / "git" / "lammps-fork" / "build-msevb" / "lmp"


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    partitions: int
    procs: int

    @classmethod
    def parse(cls, text: str) -> "Layout":
        p, m = text.lower().split("x")
        return cls(int(p), int(m))

    @property
    def nranks(self) -> int:
        return self.partitions * self.procs

    def __str__(self) -> str:
        return f"{self.partitions}x{self.procs}"


def lmp_path() -> Path:
    return Path(os.environ.get("LMP", DEFAULT_LMP))


def mpirun_command() -> list[str]:
    return shlex.split(os.environ.get("MPIRUN", "mpirun --oversubscribe"))


def render_setup(case: Case, extra_fix_args: str = "") -> str:
    text = case.setup_file.read_text()
    if text.count(FILE_KEYWORD) != 1:
        raise ValueError(f"{case.setup_file} must contain '{FILE_KEYWORD}' exactly once")
    if extra_fix_args:
        text = text.replace(FILE_KEYWORD, f"{FILE_KEYWORD} {extra_fix_args}")
    return text


def render_input(case: Case, layout: Layout, extra_commands: str = "", mode: str | None = None) -> str:
    worlds = " ".join(str(p) for p in range(layout.partitions))
    return "\n".join(
        [
            f"variable p world {worlds}",
            "include in.setup",
            extra_commands,
            f"include {COMMON_DIR / ((mode or case.mode) + '.lmp')}",
            "",
        ]
    )


def stage(case: Case, workdir: Path, extra_fix_args: str = "") -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for src in case.input_files():
        dst = workdir / src.name
        if not dst.exists():
            dst.symlink_to(src)
    (workdir / "in.setup").write_text(render_setup(case, extra_fix_args))


def run_timeout() -> float:
    return float(os.environ.get("MSEVB_RUN_TIMEOUT", 900))


def launch(workdir: Path, layout: Layout, input_name: str = "in.test", timeout: float | None = None) -> None:
    timeout = run_timeout() if timeout is None else timeout
    lmp = lmp_path()
    if not lmp.exists():
        raise RunError(f"LAMMPS binary not found at {lmp}; set LMP")
    cmd = mpirun_command() + [
        "-np", str(layout.nranks),
        str(lmp),
        "-partition", str(layout),
        "-in", input_name,
        "-log", "log.lammps",
        "-screen", "none",
        "-nocite",
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RunError(
            f"{shlex.join(cmd)} did not finish within {timeout:g} s in {workdir} "
            f"(possible deadlock between partitions)\n--- log tail ---\n{_log_tail(workdir)}"
        ) from exc
    if proc.returncode != 0:
        raise RunError(
            f"{shlex.join(cmd)} failed in {workdir} (exit {proc.returncode})\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}\n"
            f"--- log tail ---\n{_log_tail(workdir)}"
        )


def run_case(
    case: Case,
    layout: Layout,
    workdir: Path,
    extra_fix_args: str = "",
    extra_commands: str = "",
    mode: str | None = None,
) -> Result:
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    stage(case, workdir, extra_fix_args)
    (workdir / "in.test").write_text(render_input(case, layout, extra_commands, mode))
    launch(workdir, layout)
    try:
        return parse_run(workdir, layout.partitions, str(layout))
    except (OSError, ValueError) as exc:
        raise RunError(f"could not read outputs in {workdir}: {exc}\n{_log_tail(workdir)}") from exc


def _log_tail(workdir: Path, nlines: int = 40) -> str:
    chunks = []
    for log in sorted(workdir.glob("log.lammps*")):
        lines = log.read_text(errors="replace").splitlines()
        chunks.append(f"[{log.name}]\n" + "\n".join(lines[-nlines:]))
    return "\n".join(chunks)
