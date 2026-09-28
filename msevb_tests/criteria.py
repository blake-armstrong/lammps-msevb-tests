from dataclasses import dataclass, field

import numpy as np

from .cases import Case
from .results import EVBRecord


@dataclass
class CriterionReport:
    criterion: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        metrics = ", ".join(f"{k}={v:.6g}" for k, v in self.metrics.items())
        status = "PASS" if self.ok else "FAIL: " + "; ".join(self.failures)
        return f"{self.criterion} {status} ({metrics})"


def coupling_metrics(evb: EVBRecord, min_weight: float) -> dict:
    ham = evb.hamiltonian
    n = evb.nstates
    offdiag = evb.coupling_scale * ham[~np.eye(n, dtype=bool)] if n > 1 else np.zeros(1)
    metrics = {
        "nstates": n,
        "max_abs_coupling": float(np.max(np.abs(offdiag))),
        "coupling_lowering": float(np.min(np.diag(ham)) - evb.ground_energy),
        "mixed_states": int(np.count_nonzero(evb.amplitudes >= min_weight)),
    }
    if evb.occupations is not None:
        metrics["occupied_states"] = int(np.count_nonzero(evb.occupations >= min_weight))
    return metrics


def evaluate(
    evb: EVBRecord, criterion: str, min_coupling: float, min_lowering: float, min_weight: float
) -> CriterionReport:
    m = coupling_metrics(evb, min_weight)
    failures = []
    if m["nstates"] < 2:
        failures.append(f"only {m['nstates']} EVB state(s)")
    if m["mixed_states"] < 2:
        failures.append(f"{m['mixed_states']} state(s) with amplitude >= {min_weight}")

    if criterion == "coupling":
        if m["max_abs_coupling"] < min_coupling:
            failures.append(f"max |H_ij| = {m['max_abs_coupling']:.3g} < {min_coupling}")
        if m["coupling_lowering"] < min_lowering:
            failures.append(
                f"coupling lowers the ground state by {m['coupling_lowering']:.3g} < {min_lowering}"
            )
    elif criterion == "fermi_dirac":
        if evb.occupations is None:
            failures.append("record has no Fermi-Dirac occupations")
        elif m["occupied_states"] < 2:
            failures.append(f"{m['occupied_states']} eigenstate(s) with occupation >= {min_weight}")
    else:
        raise ValueError(f"unknown criterion {criterion!r}")

    return CriterionReport(criterion=criterion, ok=not failures, failures=failures, metrics=m)


def evaluate_case(case: Case, evb: EVBRecord) -> CriterionReport:
    return evaluate(evb, case.criterion, case.min_coupling, case.min_lowering, case.min_weight)
