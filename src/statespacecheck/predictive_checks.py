"""Predictive densities and Monte Carlo predictive checks of whole time bins.

These are extensions beyond the paper. The paper's predictive check is the
rank-based predictive p-value of each spike, computed exactly over the units by
:func:`~statespacecheck.mark_predictive_pvalue` and
:func:`~statespacecheck.event_diagnostics`. The functions here compute the
predictive density of all observations in a time bin, and a Monte Carlo p-value
from a user-supplied sampler.
"""

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import (
    DistributionArray,
    flatten_time_spatial,
    row_chunks,
    validate_distribution,
    validate_paired_distributions,
)

# Note: aggregate_over_period has been moved to periods.py as a generic utility


def predictive_density(
    state_dist: ArrayLike,
    observation_likelihood: ArrayLike,
) -> DistributionArray:
    """Compute predictive density by integrating state dist with obs likelihood.

    CRITICAL: This function normalizes state_dist ONLY, NOT likelihood.
    The likelihood p(y|x) is a likelihood function, not a distribution over x.
    Normalizing it over x would change its value and mask real model misfit.

    Formula: f_predictive(y) = ∑_x p(x) * p(y|x)

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_time, ...)
        State probability distributions over position at each time point where
        ... represents arbitrary spatial dimensions.
        Non-negative values (NaN allowed to mark invalid bins).
        Will be normalized over spatial dimensions (everything except time).
    observation_likelihood : np.ndarray, shape (n_time, ...)
        Observation likelihood p(y|x) of the observed data at each position.
        Non-negative values (NaN allowed to mark invalid bins).
        Not normalized over positions (unlike the ``likelihood`` of
        :func:`~statespacecheck.kl_divergence`): it is a function of x, not a
        distribution. Must have same shape as state_dist.

    Returns
    -------
    predictive_density : np.ndarray, shape (n_time,)
        Predictive density at each time point.

    Raises
    ------
    ValueError
        If state_dist and observation_likelihood have different shapes, or if
        they contain negative values.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import predictive_density
    >>> # Simple 1D example with unnormalized state
    >>> state = np.array([[3.0, 4.0, 3.0]])  # Unnormalized (sums to 10, not 1)
    >>> like = np.array([[2.0, 3.0, 1.0]])  # Likelihood values (not normalized)
    >>> pred = predictive_density(state, like)
    >>> pred.shape
    (1,)

    See Also
    --------
    log_predictive_density : Compute log predictive density for numerical stability
    kl_divergence : Measure information divergence between distributions
    hpd_overlap : Compute spatial overlap between HPD regions

    Notes
    -----
    The predictive density is computed via discrete Riemann sum:
        f_predictive(y_k) = ∑_x p(x_k) * p(y_k | x_k)

    Where:
    - p(x_k) is the state distribution (normalized to sum to 1)
    - p(y_k | x_k) is the observation likelihood (NOT normalized)

    Distributions are validated using validate_paired_distributions:
    - NaN/inf values in input are converted to 0.0
    - Shape and non-negativity are checked
    - State distribution is normalized after validation
    - Likelihood is NOT normalized (critical for correct results)

    Integration is performed by flattening spatial dimensions and computing
    row-wise sums over all spatial bins.
    """
    state = np.asarray(state_dist, dtype=float)
    like = np.asarray(observation_likelihood, dtype=float)
    if state.ndim < 2 or state.shape != like.shape:
        # Raise the usual error, which reports both full shapes
        validate_paired_distributions(
            state, like, name1="state_dist", name2="observation_likelihood", min_ndim=2
        )
    predictive: DistributionArray = np.empty(state.shape[0])
    any_zero_rows = False
    for rows in row_chunks(state.shape):
        predictive[rows], zero_rows = _predictive_density_rows(state[rows], like[rows])
        any_zero_rows |= zero_rows
    if any_zero_rows:
        warnings.warn(
            "state_dist has zero-sum rows; predictive set to NaN for those rows",
            UserWarning,
            stacklevel=2,
        )
    return predictive


def log_predictive_density(
    state_dist: ArrayLike,
    observation_likelihood: ArrayLike | None = None,
    *,
    log_observation_likelihood: ArrayLike | None = None,
) -> DistributionArray:
    """Compute log predictive density directly in log-space using logsumexp.

    CRITICAL: This function normalizes state_dist ONLY, NOT likelihood.
    Computes log predictive density natively in log-space for numerical stability.
    DO NOT compute as np.log(predictive_density(...)) - this loses precision.

    Formula: log f_predictive(y) = log ∑_x p(x) * p(y|x)
             = logsumexp(log p(x) + log p(y|x))

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_time, ...)
        State probability distributions over position at each time point where
        ... represents arbitrary spatial dimensions.
        Non-negative values (NaN allowed to mark invalid bins).
        Will be normalized over spatial dimensions (everything except time).
    observation_likelihood : np.ndarray, shape (n_time, ...), optional
        Observation likelihood p(y|x) of the observed data at each position.
        Non-negative values (NaN allowed to mark invalid bins). Not normalized
        over positions. Must have same shape as state_dist.
        Exactly one of `observation_likelihood` or `log_observation_likelihood`
        must be provided.
    log_observation_likelihood : np.ndarray, shape (n_time, ...), optional
        Log observation likelihood log p(y|x), keyword-only. Passing it avoids
        an exp/log round-trip. Must have same shape as state_dist.

    Returns
    -------
    log_predictive_density : np.ndarray, shape (n_time,)
        Log predictive density at each time point.

    Raises
    ------
    ValueError
        If neither or both of the likelihood arguments are provided,
        if shapes don't match, or if distributions contain negative values.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import log_predictive_density
    >>> state = np.array([[1.0, 1.0, 1.0]])
    >>> like = np.array([[2.0, 3.0, 4.0]])
    >>> log_pred = log_predictive_density(state, like)
    >>> log_pred.shape
    (1,)

    >>> # From a log likelihood, for numerical stability
    >>> log_pred2 = log_predictive_density(state, log_observation_likelihood=np.log(like))
    >>> np.allclose(log_pred, log_pred2)
    True

    See Also
    --------
    predictive_density : Compute predictive density in linear space
    kl_divergence : Measure information divergence between distributions
    hpd_overlap : Compute spatial overlap between HPD regions

    Notes
    -----
    This function computes log predictive density directly in log-space using
    scipy.special.logsumexp for numerical stability. This prevents underflow
    when working with very small probabilities or peaked distributions.

    The computation is:
        log ∑_x p(x) * p(y|x) = logsumexp(log p(x) + log p(y|x))

    Where:
    - p(x) is the state distribution (normalized to sum to 1)
    - p(y|x) is the observation likelihood (NOT normalized)

    For users who already have the log likelihood, passing it via
    `log_observation_likelihood` avoids the exp/log round-trip and is more
    efficient and numerically stable.
    """
    if (observation_likelihood is None) == (log_observation_likelihood is None):
        msg = (
            "Exactly one of 'observation_likelihood' or 'log_observation_likelihood' "
            "must be provided"
        )
        raise ValueError(msg)

    state = np.asarray(state_dist, dtype=float)
    if observation_likelihood is not None:
        like = np.asarray(observation_likelihood, dtype=float)
        if state.ndim < 2 or state.shape != like.shape:
            # Raise the usual error, which reports both full shapes
            validate_paired_distributions(
                state, like, name1="state_dist", name2="observation_likelihood", min_ndim=2
            )
    else:
        # Validate the log likelihood manually (it's in log-space, can be negative!)
        like = np.asarray(log_observation_likelihood, dtype=float)
        if like.ndim < 2:
            msg = (
                f"log_observation_likelihood must be at least 2D with shape (n_time, ...), "
                f"got shape {like.shape}"
            )
            raise ValueError(msg)
        if state.ndim < 2:
            validate_distribution(state, name="state_dist", min_ndim=2)
        if like.shape != state.shape:
            msg = (
                f"state_dist and log_observation_likelihood must have same shape, "
                f"got {state.shape} vs {like.shape}"
            )
            raise ValueError(msg)
        # +inf in the log likelihood indicates an upstream bug or overflow
        if np.isposinf(like).any():
            msg = (
                "log_observation_likelihood contains +inf; this indicates an upstream "
                "bug or overflow"
            )
            raise ValueError(msg)

    log_predictive: DistributionArray = np.empty(state.shape[0])
    any_zero_rows = False
    for rows in row_chunks(state.shape):
        log_predictive[rows], zero_rows = _log_predictive_density_rows(
            state[rows], like[rows], is_log=observation_likelihood is None
        )
        any_zero_rows |= zero_rows
    if any_zero_rows:
        warnings.warn(
            "state_dist has zero-sum rows; predictive set to NaN for those rows",
            UserWarning,
            stacklevel=2,
        )
    return log_predictive


def predictive_pvalue(
    observed_log_pred: ArrayLike,
    sample_log_pred: Callable[[int], ArrayLike],
    *,
    n_samples: int = 1000,
) -> DistributionArray:
    """Compute predictive p-value via Monte Carlo sampling.

    Computes p-values for predictive checks by comparing observed log predictive
    densities to a distribution of simulated log predictive densities. The p-value
    at each time point is the proportion of simulated values that are less than
    or equal to the observed value.

    This computes a Monte Carlo predictive check p-value for a user-supplied
    replicate-generating procedure. If the model is correct and the statistic is
    continuous, p-values should be approximately uniformly distributed; ties and
    discreteness can make them conservative. Systematic deviations indicate model
    misfit.

    Parameters
    ----------
    observed_log_pred : np.ndarray, shape (n_time,)
        Observed log predictive densities for actual data.
        Must be 1-dimensional.
    sample_log_pred : callable
        Function that generates samples of log predictive densities under the model.
        Must accept a single integer argument `n_samples` and return an array of
        shape (n_samples, n_time) containing simulated log predictive densities.
        For reproducibility, use np.random.Generator with a fixed seed internally.
        Example: `lambda n: rng.normal(loc=model_mean, scale=model_std, size=(n, n_time))`
    n_samples : int, optional
        Number of Monte Carlo samples to draw for p-value computation.
        Higher values give more accurate p-value estimates but take longer.
        Default is 1000.

    Returns
    -------
    p_values : np.ndarray, shape (n_time,)
        P-value at each time point, computed as the proportion of simulated
        log predictive densities <= observed value.
        Values range from 0 to 1; NaN where ``observed_log_pred`` is NaN.

    Raises
    ------
    ValueError
        If observed_log_pred is not 1-dimensional, if n_samples <= 0,
        or if sample_log_pred returns an array with the wrong shape or
        containing NaN.
    TypeError
        If sample_log_pred is not callable.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import predictive_pvalue
    >>> # Observed log predictive densities
    >>> observed = np.array([-2.0, -1.5, -1.0])
    >>> # Sampler with internal random state for reproducibility
    >>> def sampler(n_samples):
    ...     rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    ...     return rng.normal(loc=-1.5, scale=0.5, size=(n_samples, 3))
    >>> # Monte Carlo estimates of the exact values 0.16, 0.5 and 0.84
    >>> predictive_pvalue(observed, sampler, n_samples=1000).round(2)
    array([0.16, 0.5 , 0.86])

    See Also
    --------
    log_predictive_density : Compute log predictive density for observed data
    predictive_density : Compute predictive density in linear space
    aggregate_over_period : Aggregate metrics over time periods

    Notes
    -----
    The p-value at time t is computed as:
        p_value[t] = (1 / n_samples) * sum(simulated[t] <= observed[t])

    Interpretation: the statistic is a log predictive density, so a small
    p-value means the observed data were less probable than nearly all
    replicates, i.e. unexpected under the model. A p-value near 1 means the
    observation was among the most probable outcomes, which is good fit, not
    misfit. Flag small values (for example ``p <= 0.05``, as
    :func:`~statespacecheck.periods.flag_extreme_pvalues` does).

    The estimate is the fraction of ``n_samples`` replicates, so it is a
    multiple of ``1 / n_samples`` and can be exactly 0; choose ``n_samples``
    large enough to resolve the cutoff you use.

    The sampler function should:
    1. Generate new data from the model
    2. Compute log predictive density for each generated dataset
    3. Return array of shape (n_samples, n_time)
    4. Use np.random.Generator internally for reproducibility

    For reproducible results, create your sampler with a fixed seed:
        rng = np.random.default_rng(42)
        sampler = lambda n: rng.normal(size=(n, n_time))
    """
    # Validate observed_log_pred
    observed_arr = np.asarray(observed_log_pred, dtype=float)
    if observed_arr.ndim != 1:
        msg = (
            f"observed_log_pred must be 1-dimensional, "
            f"got {observed_arr.ndim}D array with shape {observed_arr.shape}"
        )
        raise ValueError(msg)

    n_time = observed_arr.shape[0]

    # Validate n_samples
    if n_samples <= 0:
        msg = f"n_samples must be positive, got {n_samples}"
        raise ValueError(msg)

    # Generate samples
    simulated = sample_log_pred(n_samples)

    # Validate shape of simulated samples
    simulated_arr = np.asarray(simulated, dtype=float)
    if simulated_arr.shape != (n_samples, n_time):
        msg = (
            f"sample_log_pred output must have shape (n_samples, n_time) = "
            f"({n_samples}, {n_time}), got shape {simulated_arr.shape}"
        )
        raise ValueError(msg)
    # A NaN sample compares False, which would silently pull the p-value toward 0
    # and read as misfit; it is a sampler error.
    nan_times = np.isnan(simulated_arr).any(axis=0)
    if nan_times.any():
        bad = np.flatnonzero(nan_times)
        msg = f"sample_log_pred returned NaN at time indices: {bad[:10].tolist()}"
        raise ValueError(msg)

    # Proportion of samples <= observed at each time: (n_samples, n_time) -> (n_time,).
    # A NaN observation gives a NaN p-value; +-inf follow the comparison
    # (-inf, impossible under the model, gives 0).
    mask = ~np.isnan(observed_arr)
    p_values: DistributionArray = np.full(n_time, np.nan)
    if np.any(mask):
        p_values[mask] = np.mean(simulated_arr[:, mask] <= observed_arr[mask], axis=0)
    return p_values


def _predictive_density_rows(
    state_dist: DistributionArray, observation_likelihood: DistributionArray
) -> tuple[DistributionArray, bool]:
    """Compute :func:`predictive_density` for one chunk; also report zero-sum rows."""
    # Validate both distributions (converts NaN/inf to 0, checks shapes)
    state, like = validate_paired_distributions(
        state_dist,
        observation_likelihood,
        name1="state_dist",
        name2="observation_likelihood",
        min_ndim=2,
    )

    # Flatten for vectorized operations
    state_flat = flatten_time_spatial(state)
    like_flat = flatten_time_spatial(like)

    # Normalize state distribution ONLY (not likelihood!)
    # Shape: (n_time,)
    state_sum = state_flat.sum(axis=1)

    # Check for zero-sum state rows before normalization
    zero_rows = state_sum == 0

    # Normalize state, handling zero-sum rows
    with np.errstate(divide="ignore", invalid="ignore"):
        state_normalized = state_flat / state_sum[:, np.newaxis]

    # Replace non-finite values (from zero-sum rows) with 0
    state_normalized = np.nan_to_num(state_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute predictive density: sum over spatial dimensions
    # f_predictive(y) = ∑_x p(x) * p(y|x)
    # Note: likelihood is NOT normalized (critical!)
    predictive: DistributionArray = (state_normalized * like_flat).sum(axis=1)

    # Set zero-sum rows to NaN (they have no valid state mass)
    predictive[zero_rows] = np.nan

    return predictive, bool(zero_rows.any())


def _log_predictive_density_rows(
    state_dist: DistributionArray, likelihood: DistributionArray, *, is_log: bool
) -> tuple[DistributionArray, bool]:
    """Compute :func:`log_predictive_density` for one chunk; also report zero-sum rows.

    ``likelihood`` is the observation likelihood, or its log if ``is_log``.
    """
    if not is_log:
        # Validate state distribution (a probability) and likelihood (a function, not a dist)
        state, like = validate_paired_distributions(
            state_dist,
            likelihood,
            name1="state_dist",
            name2="observation_likelihood",
            min_ndim=2,
        )
        # Convert to log-space (avoiding log(0) by using where)
        like_flat = flatten_time_spatial(like)
        with np.errstate(divide="ignore"):
            log_like_flat = np.where(like_flat > 0, np.log(like_flat), -np.inf)
    else:
        state = validate_distribution(state_dist, name="state_dist", min_ndim=2)
        # NaN -> -inf (zero likelihood); negative values are expected in log-space
        log_like = np.nan_to_num(likelihood, nan=-np.inf, neginf=-np.inf)
        log_like_flat = flatten_time_spatial(log_like)

    # Flatten state for vectorized operations
    state_flat = flatten_time_spatial(state)

    # Normalize state distribution ONLY (not likelihood!)
    state_sum = state_flat.sum(axis=1)

    # Check for zero-sum state rows before normalization
    zero_rows = state_sum == 0

    # Normalize state, handling zero-sum rows
    with np.errstate(divide="ignore", invalid="ignore"):
        state_normalized = state_flat / state_sum[:, np.newaxis]

    # Replace non-finite values (from zero-sum rows) with 0
    state_normalized = np.nan_to_num(state_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert normalized state to log-space
    with np.errstate(divide="ignore"):
        log_state_normalized = np.where(
            state_normalized > 0, np.log(state_normalized), -np.inf
        )

    # Compute log predictive density using logsumexp
    # log ∑_x p(x) * p(y|x) = logsumexp(log p(x) + log p(y|x))
    log_predictive: DistributionArray = logsumexp(log_state_normalized + log_like_flat, axis=1)

    # Set zero-sum rows to NaN (they have no valid state mass)
    log_predictive[zero_rows] = np.nan

    return log_predictive, bool(zero_rows.any())
