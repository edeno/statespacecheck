"""Rows with extreme total mass (subnormal or overflowing), rows with no bins, and
per-event diagnostics at extreme input scales."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from statespacecheck import (
    event_diagnostics,
    highest_density_region,
    hpd_overlap,
    kl_divergence,
    log_predictive_density,
    predictive_density,
)


class TestKLDivergenceSubnormalNumbers:
    """Test KL divergence handles subnormal numbers correctly."""

    def test_kl_divergence_with_subnormal_values_is_non_negative(self):
        """Test that KL divergence with subnormal values doesn't return negative.

        This test uses the exact failing case found by Hypothesis that triggers
        floating point precision errors in scipy.stats.entropy, which can return
        tiny negative values (~10^-113) instead of 0 for nearly identical distributions.

        KL divergence is mathematically always non-negative. The implementation
        must clip spurious negative floating point artifacts to ensure this property.
        """
        # Exact failing case from Hypothesis property test
        dist1 = np.array([[4.39835706e-113, 2.00000000e000]])
        dist2 = np.array([[4.39835706e-113, 1.00000000e000]])

        kl_div = kl_divergence(dist1, dist2)

        # KL divergence must be non-negative (mathematical property)
        # No tolerance - must be exactly >= 0 since we clip in implementation
        assert kl_div[0] >= 0.0, (
            f"KL divergence must be non-negative, got {kl_div[0]:.20e}. "
            "This indicates floating point precision issues with subnormal numbers."
        )

    def test_kl_divergence_clips_tiny_negative_to_zero(self):
        """Test that tiny negative values from floating point errors are clipped to 0."""
        # Another case that might trigger similar issues
        dist1 = np.array([[1e-200, 1.0]])
        dist2 = np.array([[2e-200, 1.0]])

        kl_div = kl_divergence(dist1, dist2)

        # Should be non-negative
        assert kl_div[0] >= 0.0

        # Should be very close to 0 (distributions are nearly identical except for tiny values)
        assert kl_div[0] < 1e-10


# Each takes rows of nonnegative values; results must not depend on their overall scale
ROW_COMPUTATIONS = {
    "kl_divergence": lambda s: kl_divergence(s, s[:, ::-1]),
    "hpd_overlap": lambda s: hpd_overlap(s, s[:, ::-1]),
    "highest_density_region": lambda s: highest_density_region(s).astype(float),
    "predictive_density": lambda s: predictive_density(s, np.arange(1.0, 10.0)[None]),
    "log_predictive_density": lambda s: log_predictive_density(s, np.arange(1.0, 10.0)[None]),
}


@pytest.mark.parametrize("name", ROW_COMPUTATIONS)
def test_row_with_subnormal_total_mass(name) -> None:
    """A row whose total mass is subnormal gives the same result as the row scaled
    up; dividing by the subnormal total warned about an overflow on NumPy 1.26."""
    tiny = np.zeros((1, 9))
    tiny[0, [0, 3, 5]] = [1.0e-309, 1.2e-309, 0.3e-309]  # 1 / sum exceeds the largest float
    compute = ROW_COMPUTATIONS[name]
    np.testing.assert_allclose(compute(tiny), compute(tiny * 2.0**1000), rtol=1e-12)


@pytest.mark.parametrize("name", ROW_COMPUTATIONS)
def test_row_whose_total_mass_overflows(name) -> None:
    """A row of finite values whose sum overflows gives the same result as the row
    scaled down, instead of infinite KL divergence or an empty region."""
    huge = np.zeros((1, 9))
    huge[0, [0, 3, 5]] = [1.0e308, 1.2e308, 0.3e308]
    compute = ROW_COMPUTATIONS[name]
    np.testing.assert_allclose(compute(huge), compute(huge * 2.0**-1000), rtol=1e-12)


@pytest.mark.parametrize(
    "compute",
    [
        lambda s: kl_divergence(s, s),
        lambda s: hpd_overlap(s, s),
        lambda s: highest_density_region(s),
        lambda s: predictive_density(s, s),
        lambda s: log_predictive_density(s, s),
    ],
    ids=["kl", "hpd", "hdr", "predictive", "log_predictive"],
)
def test_empty_spatial_axis_raises(compute) -> None:
    with pytest.raises(ValueError, match="no bins"):
        compute(np.ones((3, 0)))


def test_event_diagnostics_with_underflowing_intensity_products():
    """Products below the smallest subnormal underflow to 0, but the events have
    positive intensity; the diagnostics use the exact probabilities (1/3, 2/3)."""
    u = np.finfo(float).smallest_subnormal
    state = np.array([[0.125, 0.875]])
    rates = np.array([[u, 2 * u], [0.0, 0.0]])
    diagnostics = event_diagnostics(state, rates, np.array([0, 0]), np.array([0, 1]))
    assert_allclose(diagnostics.predictive_pvalue, [1 / 3, 1.0])
    assert_array_equal(diagnostics.kl_divergence, [np.inf, np.inf])


@pytest.fixture(scope="module")
def sweep_model():
    """A model with an exact tie (marks 0 and 1), a permuted copy (mark 2), zeros,
    and an impossible event (mark 5 at time bin 0), with every (time bin, mark)
    pair as an event."""
    rng = np.random.default_rng(0)
    n_time, n_bins, n_marks = 5, 12, 6
    state = rng.random((n_time, n_bins)) * (rng.random((n_time, n_bins)) < 0.8)
    state[:, 0] = np.maximum(state[:, 0], 0.1)
    rates = rng.random((n_bins, n_marks)) * (rng.random((n_bins, n_marks)) < 0.7)
    rates[:, 1] = rates[:, 0]
    rates[:, 2] = rates[::-1, 0]
    rates[:, 5] = 0.0
    rates[n_bins - 1, 5] = 0.5
    state[0, n_bins - 1] = 0.0
    rates[0, :] = np.maximum(rates[0, :], 0.05)
    time_ind, marks = (
        a.ravel() for a in np.meshgrid(range(n_time), range(n_marks), indexing="ij")
    )
    return state, rates, time_ind, marks


@pytest.mark.parametrize(
    ("state_exponent", "rates_exponent"),
    [(0, -1060), (-1040, 0), (-700, -700), (-300, -1000), (-1040, -1060), (500, 300)],
)
def test_event_diagnostics_do_not_depend_on_the_scale_of_the_inputs(
    sweep_model, state_exponent, rates_exponent
):
    """Scaling the state by 2**a and the rates by 2**b (into or below the subnormal
    range) gives the diagnostics of the same stored numbers scaled back, an exact
    power-of-two reference: the same model at ordinary scale."""
    state, rates, time_ind, marks = sweep_model
    small_state = np.ldexp(state, state_exponent)
    small_rates = np.ldexp(rates, rates_exponent)
    small = event_diagnostics(small_state, small_rates, time_ind, marks)
    reference = event_diagnostics(
        np.ldexp(small_state, -state_exponent),
        np.ldexp(small_rates, -rates_exponent),
        time_ind,
        marks,
    )
    for name in ("predictive_pvalue", "hpd_overlap", "kl_divergence"):
        assert_allclose(getattr(small, name), getattr(reference, name), rtol=1e-11, atol=0)
