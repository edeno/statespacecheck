"""The time-bin functions process time in chunks; results must not depend on it."""

import contextlib

import numpy as np
import pytest

import statespacecheck._validation
from statespacecheck import (
    highest_density_region,
    hpd_overlap,
    kl_divergence,
    log_predictive_density,
    predictive_density,
)


@pytest.fixture
def pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    state = rng.dirichlet(np.ones(12) * 0.5, size=40).reshape(40, 3, 4)
    like = rng.gamma(1.0, size=(40, 3, 4))
    state[5] = 0.0  # a zero-mass row
    state[7, 0, 0] = np.nan  # an invalid bin
    return state, like


# (computation, whether it warns about the zero-mass row)
COMPUTATIONS = {
    "kl_divergence": (kl_divergence, False),
    "hpd_overlap": (hpd_overlap, False),
    "highest_density_region": (lambda s, _: highest_density_region(s), False),
    "predictive_density": (predictive_density, True),
    "log_predictive_density": (log_predictive_density, True),
    "log_predictive_density_log": (
        lambda s, lk: log_predictive_density(s, log_observation_likelihood=np.log(lk)),
        True,
    ),
}


@pytest.mark.parametrize("name", COMPUTATIONS)
def test_results_do_not_depend_on_chunk_size(monkeypatch, pair, name):
    compute, warns = COMPUTATIONS[name]

    def run() -> np.ndarray:
        context = (
            pytest.warns(UserWarning, match="zero-sum rows")
            if warns
            else contextlib.nullcontext()
        )
        with context:
            return compute(*pair)

    whole = run()
    # 3 rows per chunk, so the last of the 40 rows is a partial chunk of one row
    monkeypatch.setattr(statespacecheck._validation, "_CHUNK_ELEMENTS", 36)
    np.testing.assert_array_equal(run(), whole)


def test_zero_mass_warning_is_emitted_once(monkeypatch, pair):
    monkeypatch.setattr(statespacecheck._validation, "_CHUNK_ELEMENTS", 25)
    state, like = pair
    state[30] = 0.0  # a second zero-mass row, in another chunk
    with pytest.warns(UserWarning, match="zero-sum rows") as record:
        predictive_density(state, like)
    assert len(record) == 1
