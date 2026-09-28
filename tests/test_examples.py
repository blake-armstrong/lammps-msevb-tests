import numpy as np
import pytest

from conftest import cached_run
from msevb_tests.cases import find_case
from msevb_tests.compare import assert_close, tolerance
from msevb_tests.criteria import coupling_metrics, evaluate_case
from msevb_tests.golden import load_golden
from msevb_tests.runner import Layout
from suite import (
    test_energy_matches_golden,
    test_evb_matches_golden,
    test_forces_match_golden,
    test_golden_satisfies_criterion,
    test_partitions_identical,
    test_run_satisfies_criterion,
)

GROUP = "examples"
pytestmark = pytest.mark.examples


def test_criterion_detects_zero_coupling(run_cache, tmp_path_factory):
    case = find_case("examples/h3o_grimme2015")
    golden = load_golden(case)
    result = cached_run(case, Layout(1, 1), run_cache, tmp_path_factory, "coupling_scale 0.0")

    report = evaluate_case(case, result.evb)
    assert not report.ok, f"criterion passed with the coupling switched off: {report.summary()}"
    assert report.metrics["max_abs_coupling"] == 0.0

    lowering = coupling_metrics(golden.evb, case.min_weight)["coupling_lowering"]
    assert_close("pe rise without coupling", result.pe[0] - golden.pe, lowering, **tolerance(case, "energy"))
    force_change = np.abs(result.atoms[0].forces - golden.atoms.forces).max()
    assert force_change > 1.0e-2, f"switching the coupling off changed forces by only {force_change:.3g}"
