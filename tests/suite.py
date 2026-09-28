import numpy as np
import pytest

from msevb_tests.compare import assert_close, tolerance
from msevb_tests.criteria import evaluate_case
from msevb_tests.golden import compare_energy, compare_evb, compare_forces


def test_golden_satisfies_criterion(case, golden):
    report = evaluate_case(case, golden.evb)
    assert report.ok, report.summary()


def test_run_satisfies_criterion(case, result):
    report = evaluate_case(case, result.evb)
    assert report.ok, report.summary()


def test_energy_matches_golden(case, result, golden):
    compare_energy(case, result, golden)


def test_evb_matches_golden(case, result, golden):
    compare_evb(case, result, golden)


def test_forces_match_golden(case, result, golden):
    compare_forces(case, result.atoms[0], golden)


def test_partitions_identical(case, result):
    if len(result.atoms) == 1:
        pytest.skip("single partition")
    ref = result.atoms[0]
    for p in range(1, len(result.atoms)):
        assert_close(f"pe on partition {p}", result.pe[p], result.pe[0], **tolerance(case, "energy"))
        atoms = result.atoms[p]
        assert np.array_equal(atoms.ids, ref.ids), f"atom ids differ on partition {p}"
        assert_close(f"forces on partition {p}", atoms.forces, ref.forces, **tolerance(case, "forces"))
