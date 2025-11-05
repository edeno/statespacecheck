"""Predictive check functions for state space model goodness of fit.

This module provides functions to compute predictive densities and
perform predictive checks for Bayesian state space models.
"""

import warnings
from collections.abc import Callable

import numpy as np
from scipy.special import logsumexp

from ._validation import (
    DistributionArray,
    flatten_time_spatial,
    validate_distribution,
    validate_paired_distributions,
)

# Note: aggregate_over_period has been moved to periods.py as a generic utility


def predictive_density(
    state_dist: DistributionArray,
    likelihood: DistributionArray,
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
    likelihood : np.ndarray, shape (n_time, ...)
        Likelihood p(y|x) evaluated at observed data across all positions.
        Non-negative values (NaN allowed to mark invalid bins).
        DO NOT normalize - this is a likelihood function.
        Must have same shape as state_dist.

    Returns
    -------
    predictive_density : np.ndarray, shape (n_time,)
        Predictive density at each time point.

    Raises
    ------
    ValueError
        If state_dist and likelihood have different shapes, or if distributions
        contain negative values.

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
    # Validate both distributions (converts NaN/inf to 0, checks shapes)
    state, like = validate_paired_distributions(
        state_dist, likelihood, name1="state_dist", name2="likelihood", min_ndim=2
    )

    # Flatten for vectorized operations
    state_flat = flatten_time_spatial(state)
    like_flat = flatten_time_spatial(like)

    # Normalize state distribution ONLY (not likelihood!)
    # Shape: (n_time,)
    state_sum = state_flat.sum(axis=1)

    # Check for zero-sum state rows before normalization
    zero_rows = state_sum == 0
    if np.any(zero_rows):
        warnings.warn(
            "state_dist has zero-sum rows; predictive set to NaN for those rows",
            UserWarning,
            stacklevel=2,
        )

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

    return predictive


def log_predictive_density(
    state_dist: DistributionArray,
    likelihood: DistributionArray | None = None,
    log_likelihood: DistributionArray | None = None,
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
    likelihood : np.ndarray, shape (n_time, ...), optional
        Likelihood p(y|x) evaluated at observed data across all positions.
        Non-negative values (NaN allowed to mark invalid bins).
        DO NOT normalize - this is a likelihood function.
        Must have same shape as state_dist.
        Exactly one of `likelihood` or `log_likelihood` must be provided.
    log_likelihood : np.ndarray, shape (n_time, ...), optional
        Log-likelihood log p(y|x) evaluated at observed data.
        Allows users who already have log-likelihood to avoid exp/log round-trip.
        Must have same shape as state_dist.
        Exactly one of `likelihood` or `log_likelihood` must be provided.

    Returns
    -------
    log_predictive_density : np.ndarray, shape (n_time,)
        Log predictive density at each time point.

    Raises
    ------
    ValueError
        If neither or both of likelihood and log_likelihood are provided,
        if shapes don't match, or if distributions contain negative values.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import log_predictive_density
    >>> # Using likelihood
    >>> state = np.array([[1.0, 1.0, 1.0]])
    >>> like = np.array([[2.0, 3.0, 4.0]])
    >>> log_pred = log_predictive_density(state, likelihood=like)
    >>> log_pred.shape
    (1,)

    >>> # Using log_likelihood for numerical stability
    >>> log_like = np.log(like)
    >>> log_pred2 = log_predictive_density(state, log_likelihood=log_like)
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

    For users who already have log-likelihood computed, passing it via
    `log_likelihood` parameter avoids the exp/log round-trip and is more
    efficient and numerically stable.
    """
    # Validate that exactly one of likelihood or log_likelihood is provided
    if (likelihood is None) == (log_likelihood is None):
        raise ValueError("Exactly one of 'likelihood' or 'log_likelihood' must be provided")

    # Convert likelihood to log_likelihood if needed
    if likelihood is not None:
        # Validate both distributions (both are probabilities)
        state, like = validate_paired_distributions(
            state_dist, likelihood, name1="state_dist", name2="likelihood", min_ndim=2
        )
        # Convert to log-space (avoiding log(0) by using where)
        like_flat = flatten_time_spatial(like)
        with np.errstate(divide="ignore"):
            log_like_flat = np.where(like_flat > 0, np.log(like_flat), -np.inf)
    else:
        # Validate state_dist (it's a probability)
        state = validate_distribution(state_dist, name="state_dist", min_ndim=2)

        # Validate log_likelihood manually (it's in log-space, can be negative!)
        log_like = np.asarray(log_likelihood, dtype=float)

        if log_like.ndim < 2:
            raise ValueError(
                f"log_likelihood must be at least 2D with shape (n_time, ...), "
                f"got shape {log_like.shape}"
            )

        if log_like.shape != state.shape:
            raise ValueError(
                f"state_dist and log_likelihood must have same shape, "
                f"got {state.shape} vs {log_like.shape}"
            )

        # Check for +inf in log_likelihood (indicates upstream bug or overflow)
        if np.isposinf(log_like).any():
            raise ValueError(
                "log_likelihood contains +inf; this indicates an upstream bug or overflow"
            )

        # Handle non-finite values: NaN → -inf (makes sense in log-space)
        # Note: We do NOT check for negative values (negative is expected in log-space!)
        log_like = np.nan_to_num(log_like, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

        log_like_flat = flatten_time_spatial(log_like)

    # Flatten state for vectorized operations
    state_flat = flatten_time_spatial(state)

    # Normalize state distribution ONLY (not likelihood!)
    state_sum = state_flat.sum(axis=1)

    # Check for zero-sum state rows before normalization
    zero_rows = state_sum == 0
    if np.any(zero_rows):
        warnings.warn(
            "state_dist has zero-sum rows; predictive set to NaN for those rows",
            UserWarning,
            stacklevel=2,
        )

    # Normalize state, handling zero-sum rows
    with np.errstate(divide="ignore", invalid="ignore"):
        state_normalized = state_flat / state_sum[:, np.newaxis]

    # Replace non-finite values (from zero-sum rows) with 0
    state_normalized = np.nan_to_num(state_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert normalized state to log-space
    with np.errstate(divide="ignore"):
        log_state_normalized = np.where(state_normalized > 0, np.log(state_normalized), -np.inf)

    # Compute log predictive density using logsumexp
    # log ∑_x p(x) * p(y|x) = logsumexp(log p(x) + log p(y|x))
    log_predictive: DistributionArray = logsumexp(log_state_normalized + log_like_flat, axis=1)

    # Set zero-sum rows to NaN (they have no valid state mass)
    log_predictive[zero_rows] = np.nan

    return log_predictive


def predictive_pvalue(
    observed_log_pred: DistributionArray,
    sample_log_pred: Callable[[int], DistributionArray],
    *,
    n_samples: int = 1000,
) -> DistributionArray:
    """Compute predictive p-value via Monte Carlo sampling.

    Computes p-values for predictive checks by comparing observed log predictive
    densities to a distribution of simulated log predictive densities. The p-value
    at each time point is the proportion of simulated values that are greater than
    or equal to the observed value.

    This provides a posterior predictive check: if the model is correct, p-values
    should be uniformly distributed. Systematic deviations indicate model misfit.

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
        log predictive densities >= observed value.
        Values range from 0 to 1.

    Raises
    ------
    ValueError
        If observed_log_pred is not 1-dimensional, if n_samples <= 0,
        or if sample_log_pred returns array with wrong shape.
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
    >>> p_vals = predictive_pvalue(observed, sampler, n_samples=100)
    >>> p_vals.shape
    (3,)
    >>> np.all((p_vals >= 0) & (p_vals <= 1))
    True

    See Also
    --------
    log_predictive_density : Compute log predictive density for observed data
    predictive_density : Compute predictive density in linear space
    aggregate_over_period : Aggregate metrics over time periods

    Notes
    -----
    The p-value at time t is computed as:
        p_value[t] = (1 / n_samples) * sum(simulated[t] >= observed[t])

    Interpretation:
    - p-value near 0.5: observed data consistent with model
    - p-value near 0 or 1: observed data extreme relative to model predictions
    - Systematic patterns across time suggest model misspecification

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

    # Validate sample_log_pred is callable
    if not callable(sample_log_pred):
        msg = f"sample_log_pred must be callable, got {type(sample_log_pred).__name__}"
        raise TypeError(msg)

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

    # Compute p-values: proportion of samples >= observed
    # Broadcasting: observed_arr has shape (n_time,), simulated_arr has shape (n_samples, n_time)
    # Comparison broadcasts to (n_samples, n_time), then mean over axis=0 gives (n_time,)
    # Explicit type annotation for mypy strict mode
    p_values: DistributionArray = np.mean(simulated_arr >= observed_arr, axis=0)
    return p_values
