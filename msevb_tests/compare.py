from dataclasses import dataclass

import numpy as np

from .cases import Case

DEFAULT_TOLERANCES = {
    "energy": {"rtol": 1.0e-10, "atol": 1.0e-7},
    "hamiltonian": {"rtol": 1.0e-10, "atol": 1.0e-7},
    "amplitudes": {"rtol": 0.0, "atol": 1.0e-7},
    "forces": {"rtol": 1.0e-7, "atol": 1.0e-7},
}


def tolerance(case: Case, quantity: str) -> dict:
    return {**DEFAULT_TOLERANCES[quantity], **case.tolerances.get(quantity, {})}


@dataclass
class Deviation:
    max_abs: float
    max_rel: float
    worst_ratio: float
    worst_index: tuple
    nbad: int
    size: int


def deviation(actual, expected, rtol: float, atol: float) -> Deviation:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise AssertionError(f"shape mismatch: got {a.shape}, expected {e.shape}")
    err = np.abs(a - e)
    limit = atol + rtol * np.abs(e)
    ratio = err / limit
    worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape) if ratio.size else ()
    rel = err / np.maximum(np.abs(e), np.finfo(float).tiny)
    return Deviation(
        max_abs=float(err.max(initial=0.0)),
        max_rel=float(rel.max(initial=0.0)),
        worst_ratio=float(ratio.max(initial=0.0)),
        worst_index=tuple(int(i) for i in worst),
        nbad=int(np.count_nonzero(err > limit)),
        size=int(err.size),
    )


def assert_close(label: str, actual, expected, rtol: float, atol: float, index_labels=None) -> Deviation:
    dev = deviation(actual, expected, rtol, atol)
    if dev.nbad:
        a = np.asarray(actual, dtype=float)[dev.worst_index]
        e = np.asarray(expected, dtype=float)[dev.worst_index]
        where = dev.worst_index
        if index_labels is not None and where:
            where = index_labels(where)
        raise AssertionError(
            f"{label}: {dev.nbad}/{dev.size} values outside rtol={rtol:g}, atol={atol:g}; "
            f"worst at {where}: got {a:.17g}, expected {e:.17g} "
            f"(|diff| = {abs(a - e):.3g}, max |diff| = {dev.max_abs:.3g})"
        )
    return dev
