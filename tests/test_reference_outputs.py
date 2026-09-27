"""Per-event diagnostics must not change: the paper's reported numbers depend on them.

HPD overlap must match exactly. KL divergence, the likelihood and the exact p-value may
differ by a few units in the last place between NumPy versions and platforms (the
order of summation differs), so they are compared to a relative tolerance far below
any real change. A Monte Carlo p-value is a fraction of replicates; a last-place
difference could move a near-tie across the observed density, so it may differ by one
replicate, far below the Monte Carlo error any real change would show.
The reference values in ``data/reference_outputs.npz`` were computed by this module's
``_compute``. Regenerate them (``python tests/test_reference_outputs.py``) only for a
change that is meant to alter these numbers.
"""

from pathlib import Path

import numpy as np
import pytest
from helpers import unit_sampler
from scipy.special import logsumexp
from scipy.stats import norm

import statespacecheck as ssc

REFERENCE = Path(__file__).parent / "data" / "reference_outputs.npz"
CLUSTERLESS_SAMPLES = 200


def _compute() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260926)
    outputs = {}

    # 1-D grid; units 0 and 1 have identical fields, so their spikes tie exactly
    predictive = rng.dirichlet(np.full(40, 0.5), size=120)
    rates = rng.gamma(2.0, size=(40, 8))
    rates[:, 1] = rates[:, 0]
    time_ind = rng.integers(0, 120, 700)
    marks = rng.integers(0, 8, 700)
    # batch_size leaves a partial last batch
    diagnostics = ssc.event_diagnostics(predictive, rates, time_ind, marks, batch_size=64)
    for field in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
        outputs[f"1d_{field}"] = getattr(diagnostics, field)

    # 2-D grid at a different coverage
    predictive_2d = rng.dirichlet(np.ones(30), size=60).reshape(60, 5, 6)
    rates_2d = rng.gamma(2.0, size=(5, 6, 5))
    diagnostics_2d = ssc.event_diagnostics(
        predictive_2d,
        rates_2d,
        rng.integers(0, 60, 200),
        rng.integers(0, 5, 200),
        coverage=0.9,
    )
    for field in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
        outputs[f"2d_{field}"] = getattr(diagnostics_2d, field)

    outputs["1d_likelihood"] = ssc.event_diagnostics(
        predictive, rates, time_ind, marks, return_likelihood=True, batch_size=64
    ).likelihood

    # Continuous marks: 6 units with Gaussian 1-D waveform amplitudes. The log
    # intensity sums over units with logsumexp, not a matrix product, whose last bits
    # can differ between BLAS libraries
    fields = 0.5 + 20.0 * np.exp(
        -0.5 * ((np.linspace(0, 1, 40)[:, None] - np.linspace(0.1, 0.9, 6)) / 0.12) ** 2
    )
    means = np.linspace(1.0, 4.0, 6)
    sample_unit = unit_sampler(fields)

    def log_intensity(marks: np.ndarray) -> np.ndarray:
        log_amplitude = norm.logpdf(np.asarray(marks)[:, :1], means, 0.3)  # (n, n_units)
        return logsumexp(log_amplitude[:, np.newaxis, :] + np.log(fields), axis=-1)

    def sample(bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(means[sample_unit(bins, rng)], 0.3)[:, np.newaxis]

    model = ssc.MarkModel(log_intensity, sample, fields.sum(axis=1))
    clusterless = ssc.clusterless_event_diagnostics(
        predictive,
        model,
        time_ind[:150],
        rng.uniform(0.5, 4.5, size=(150, 1)),
        n_samples=CLUSTERLESS_SAMPLES,
        rng=7,
        return_likelihood=True,
        batch_size=16,
    )
    for field in clusterless._fields:
        outputs[f"clusterless_{field}"] = getattr(clusterless, field)
    return outputs


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(REFERENCE) as stored:
        return dict(stored)


@pytest.fixture(scope="module")
def computed() -> dict[str, np.ndarray]:
    return _compute()


@pytest.mark.parametrize(
    "name",
    [
        f"{grid}_{field}"
        for grid in ("1d", "2d", "clusterless")
        for field in ("hpd_overlap", "kl_divergence", "predictive_pvalue")
    ]
    + ["1d_likelihood", "clusterless_likelihood"],
)
def test_matches_reference(
    computed: dict[str, np.ndarray], reference: dict[str, np.ndarray], name: str
) -> None:
    if name.endswith("hpd_overlap"):
        np.testing.assert_array_equal(computed[name], reference[name])
    elif name == "clusterless_predictive_pvalue":
        np.testing.assert_allclose(
            computed[name], reference[name], rtol=0, atol=1.5 / CLUSTERLESS_SAMPLES
        )
    else:
        np.testing.assert_allclose(computed[name], reference[name], rtol=1e-12, atol=0)


if __name__ == "__main__":
    REFERENCE.parent.mkdir(exist_ok=True)
    np.savez(REFERENCE, **_compute())
