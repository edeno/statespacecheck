"""Predictive checks of events whose marks are continuous or intractable.

A marked point-process observation model gives every event a mark (for
clusterless decoding, the event's waveform features) and a joint intensity
``lambda(x, y)`` of events with mark ``y`` at state ``x``. Its ground intensity
``Lambda(x) = integral of lambda(x, y) dy`` is the total event rate at ``x``, and
``lambda(x, y) / Lambda(x)`` is the distribution of an event's mark given the
state.

With a finite set of marks, such as the units of spike-sorted data, the
predictive check is a finite sum; use :func:`~statespacecheck.mark_predictive_pvalue`
and :func:`~statespacecheck.event_diagnostics`. When the marks are continuous or
too many to enumerate, :func:`monte_carlo_mark_pvalue` evaluates the same check
by simulation, from a :class:`MarkModel`: a function that evaluates the log of
the joint intensity of given marks at every state (:data:`LogMarkIntensity`), a
function that draws a mark for an event at a given state (:data:`MarkSampler`),
and the ground intensity. The intensity is taken as its log because densities of
many-dimensional marks are often too small to represent: the density of a mark
with 32 waveform features can be ``exp(-140)``, below the smallest float32.

State-bin indices passed to a :data:`MarkSampler` are flat indices into the
state grid, in the C order of ``state_dist.reshape(n_events, -1)``.
"""

from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ._validation import DistributionArray
from .events import (
    _validate_ground_intensity,
    _validate_state_distribution,
    event_weighted_predictive,
)

LogMarkIntensity: TypeAlias = Callable[[NDArray[Any]], NDArray[np.floating]]
"""Log of the joint intensity of marks at every state.

Called with marks of shape ``(n, *mark_shape)``; returns ``log lambda(x, y)`` for
each mark at every state bin, shape ``(n, *spatial_shape)``: finite, or ``-inf``
where the intensity is zero. Compute it in log space (for example with
``scipy.stats.norm.logpdf``); exponentiating first can underflow.
"""

MarkSampler: TypeAlias = Callable[[NDArray[np.intp], np.random.Generator], NDArray[Any]]
"""Draws one mark for an event at each given state bin.

Called with flat state-bin indices of shape ``(n,)`` and a random number
generator; returns marks of shape ``(n, *mark_shape)``, each drawn from
``lambda(x, y) / Lambda(x)`` at its bin. It must draw only from the generator it
is given, so that seeded results are reproducible.
"""


class MarkModel(NamedTuple):
    """A marked point-process observation model, as :func:`monte_carlo_mark_pvalue` uses it.

    The three parts must describe the same model: ``ground_intensity`` is the
    integral of ``exp(log_intensity)`` over marks, and ``sample`` draws from
    ``exp(log_intensity) / ground_intensity`` at each state. Bundling them keeps
    one model's parts together; it cannot check that they agree, and a mismatch
    biases the p-values.

    Attributes
    ----------
    log_intensity : LogMarkIntensity
        Log joint intensity ``log lambda(x, y)`` of marks at every state.
    sample : MarkSampler
        Draws a mark for an event at each given state bin.
    ground_intensity : np.ndarray, shape (...)
        Total event intensity ``Lambda(x)`` at every state (nonnegative).
    """

    log_intensity: LogMarkIntensity
    sample: MarkSampler
    ground_intensity: ArrayLike


# Events processed per batch in :func:`monte_carlo_mark_pvalue`. Memory is
# dominated by arrays over every replicated mark at every state, batch x
# n_samples x n_bins x 8 B (8 x 1000 x 512 x 8 B ~ 33 MB); the peak is about six
# of them (logsumexp copies its input), ~200 MB. Larger batches are not faster.
DEFAULT_MONTE_CARLO_BATCH_SIZE = 8


class MarkPredictiveCheck(NamedTuple):
    """Result of :func:`monte_carlo_mark_pvalue`.

    Attributes
    ----------
    pvalue : np.ndarray, shape (n_events,)
        Monte Carlo predictive p-value of each event's observed mark.
    observed_log_density : np.ndarray, shape (n_events,)
        Log predictive density of the observed mark, ``log f_pred(y_obs)``;
        ``-inf`` if the mark is impossible under the prediction.
    simulated_log_density : np.ndarray, shape (n_events, n_samples), or None
        Log predictive density of each replicated mark, if requested.
    """

    pvalue: DistributionArray
    observed_log_density: DistributionArray
    simulated_log_density: DistributionArray | None


def _safe_log(values: NDArray[np.floating]) -> DistributionArray:
    """Natural log, with zero giving ``-inf`` without a warning."""
    with np.errstate(divide="ignore"):
        log_values: DistributionArray = np.log(values)
    return log_values


def _check_leading_axis(values: ArrayLike, n: int, name: str) -> None:
    """Raise unless ``values`` has ``n`` entries along its first axis."""
    shape = np.shape(values)
    if not shape or shape[0] != n:
        msg = f"{name} must return one mark per state bin ({n}); got shape {shape}"
        raise ValueError(msg)


def _evaluate_log_intensity(
    log_intensity: LogMarkIntensity,
    marks: NDArray[Any],
    n: int,
    spatial_shape: tuple[int, ...],
) -> DistributionArray:
    """Evaluate ``log_intensity`` at ``n`` marks, checked, as a new ``(n, n_bins)`` array."""
    returned = log_intensity(marks)
    if isinstance(returned, np.ma.MaskedArray):
        msg = (
            "model.log_intensity returned a masked array, whose masked entries would be "
            "read as the values stored under them; return an ndarray with -inf where the "
            "intensity is zero"
        )
        raise ValueError(msg)
    dtype = np.asarray(returned).dtype
    if np.issubdtype(dtype, np.floating) and dtype.itemsize < 8:
        msg = (
            f"model.log_intensity must return float64 values; it returned {dtype}, whose "
            "rounding (about 1e-7 for float32) splits marks of equal predictive density"
        )
        raise ValueError(msg)
    values = np.array(returned, dtype=np.float64)
    if values.shape != (n, *spatial_shape):
        msg = (
            f"model.log_intensity must return shape {(n, *spatial_shape)}, the log "
            f"intensity of each of the {n} marks at every state bin; got {values.shape}"
        )
        raise ValueError(msg)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        msg = "model.log_intensity must return finite values, or -inf for zero intensity"
        raise ValueError(msg)
    return values.reshape(n, -1)


def _sample_state_bins(
    probabilities: NDArray[np.floating], n_samples: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw flat state-bin indices from each row of ``probabilities``.

    Parameters
    ----------
    probabilities : np.ndarray, shape (n_rows, n_bins)
        Rows summing to 1.
    n_samples : int
        Draws per row.
    rng : np.random.Generator
        Source of the uniform draws, one per sample.

    Returns
    -------
    bins : np.ndarray of int, shape (n_rows, n_samples)
    """
    n_rows, n_bins = probabilities.shape
    cdf = np.cumsum(probabilities, axis=1)
    cdf /= cdf[:, -1:]
    # Shifting each row's CDF by its row index makes all rows one increasing
    # sequence, so a single searchsorted draws from every row. side="right"
    # skips zero-probability bins, which add no width to the CDF.
    offsets = np.arange(n_rows)[:, np.newaxis]
    uniform = rng.random((n_rows, n_samples))
    flat = np.searchsorted((cdf + offsets).ravel(), (uniform + offsets).ravel(), side="right")
    bins = flat.reshape(n_rows, n_samples) - offsets * n_bins
    # Rounding in uniform + offset can carry a draw just below 1 past a row's last
    # bin with probability; take that bin instead
    last_with_probability = n_bins - 1 - np.argmax(probabilities[:, ::-1] > 0.0, axis=1)
    np.minimum(bins, last_with_probability[:, np.newaxis], out=bins)
    state_bins: NDArray[np.intp] = bins.astype(np.intp, copy=False)
    return state_bins


def _monte_carlo_batch(
    state: DistributionArray,
    ground: DistributionArray,
    observed_marks: NDArray[Any],
    log_intensity: LogMarkIntensity,
    sample: MarkSampler,
    spatial_shape: tuple[int, ...],
    n_samples: int,
    rng: np.random.Generator,
    first_event: int,
) -> tuple[DistributionArray, DistributionArray, DistributionArray]:
    """Monte Carlo predictive check of one batch of events.

    Parameters
    ----------
    state : np.ndarray, shape (n_batch, n_bins)
        Flattened predictive state distributions.
    ground : np.ndarray, shape (n_bins,)
        Flattened ground intensity.
    observed_marks : np.ndarray, shape (n_batch, *mark_shape)
    log_intensity, sample
        The model's functions (:class:`MarkModel`).
    spatial_shape : tuple of int
        Shape of the state grid, which ``log_intensity`` must return per mark.
    n_samples : int
        Replicated marks per event.
    rng : np.random.Generator
        Used for the state draws, then by ``sample``.
    first_event : int
        Index of the batch's first event, for error messages.

    Returns
    -------
    pvalue, observed_log, simulated_log : np.ndarray
        Shapes ``(n_batch,)``, ``(n_batch,)`` and ``(n_batch, n_samples)``.
    """
    n_batch, n_bins = state.shape
    # Lambda(x) = 0 means no events at x, so lambda(x, y) must be zero there too;
    # where the state has mass, a finite log intensity would count toward the
    # density but not toward its normalizer
    no_events = (ground == 0.0) & (state > 0.0)
    # The state's normalization cancels in the density ratio, so it is left out:
    # summing the row could overflow
    log_state = _safe_log(state)
    # log sum_x Lambda(x) P(x), the normalizer of the predictive mark density
    norm_terms = log_state + _safe_log(ground)
    log_norm = logsumexp(norm_terms, axis=1)
    no_intensity = np.flatnonzero(np.isneginf(log_norm))
    if no_intensity.size:
        msg = (
            "Event-weighted predictive distribution is undefined for rows with zero "
            f"total event intensity; row indices: {(first_event + no_intensity)[:10].tolist()}"
        )
        raise ValueError(msg)
    event_weighted = event_weighted_predictive(state, ground)

    observed_log_intensity = _evaluate_log_intensity(
        log_intensity, observed_marks, n_batch, spatial_shape
    )
    _check_zero_ground(np.isfinite(observed_log_intensity) & no_events, first_event)
    observed_terms = log_state + observed_log_intensity
    observed_sum = logsumexp(observed_terms, axis=1)

    state_bins = _sample_state_bins(event_weighted, n_samples, rng)
    replicated_marks = sample(state_bins.ravel(), rng)
    _check_leading_axis(replicated_marks, n_batch * n_samples, "model.sample")
    if np.shape(replicated_marks)[1:] != observed_marks.shape[1:]:
        msg = (
            f"model.sample must return marks of shape (n, *{observed_marks.shape[1:]}), "
            f"like observed_marks; got {np.shape(replicated_marks)}"
        )
        raise ValueError(msg)
    # The (n_batch, n_samples, n_bins) arrays dominate memory; the evaluated log
    # intensities are a new array, so the state term is added in place
    log_joint = _evaluate_log_intensity(
        log_intensity, np.asarray(replicated_marks), n_batch * n_samples, spatial_shape
    ).reshape(n_batch, n_samples, n_bins)
    if no_events.any():
        _check_zero_ground(
            (np.isfinite(log_joint) & no_events[:, np.newaxis, :]).any(axis=1), first_event
        )
    log_joint += log_state[:, np.newaxis, :]
    simulated_sum = logsumexp(log_joint, axis=2)
    # A replicate drawn at a state has positive intensity there, so its density
    # cannot be zero unless the intensity underflowed or the sampler draws marks
    # the intensity function says are impossible
    impossible = np.isneginf(simulated_sum)
    if impossible.any():
        events = first_event + np.flatnonzero(impossible.any(axis=1))
        msg = (
            f"{int(impossible.sum())} of {impossible.size} marks drawn by model.sample have "
            "zero intensity (model.log_intensity -inf) at every state with predictive "
            "mass, including the state they were drawn at (events "
            f"{events[:10].tolist()}). model.sample must draw from the model "
            "model.log_intensity describes, and model.log_intensity must be computed in "
            "log space"
        )
        raise ValueError(msg)

    observed_log = observed_sum - log_norm
    simulated_log = simulated_sum - log_norm[:, np.newaxis]
    # Marks of equal predictive density must tie. Each log density sums terms
    # log P(x) + log lambda(x, y) (and log P + log Lambda for the normalizer),
    # each rounded to about eps times its magnitude before any cancellation, so
    # the tolerance bounds those magnitudes over the terms that affect each sum,
    # plus eps per bin summed.
    magnitude = (
        _rounding_magnitude(observed_terms, log_state, observed_sum)[:, np.newaxis]
        + _rounding_magnitude(log_joint, log_state[:, np.newaxis, :], simulated_sum)
        + 2 * _rounding_magnitude(norm_terms, log_state, log_norm)[:, np.newaxis]
    )
    tolerance = 16 * np.finfo(np.float64).eps * (n_bins + magnitude)
    pvalue = np.mean(simulated_log <= observed_log[:, np.newaxis] + tolerance, axis=1)
    return pvalue, observed_log, simulated_log


def _check_zero_ground(finite_at_zero_ground: NDArray[np.bool_], first_event: int) -> None:
    """Raise if a log intensity is finite at a state with mass but no ground intensity.

    ``finite_at_zero_ground`` has shape ``(n_batch, n_bins)``.
    """
    events = np.flatnonzero(finite_at_zero_ground.any(axis=1))
    if events.size:
        msg = (
            "model.log_intensity is finite at state bins where model.ground_intensity is "
            f"zero (events {(first_event + events)[:10].tolist()}); the ground intensity "
            "must be the integral of the intensity over marks"
        )
        raise ValueError(msg)


# A term more than this far below a log sum changes it by less than exp(-40), about
# 4e-18 relative, so its own rounding does not matter
_NEGLIGIBLE_LOG_TERM = 40.0


def _rounding_magnitude(
    terms: DistributionArray, log_state: DistributionArray, total: DistributionArray
) -> DistributionArray:
    """Bound ``|log P| + |log lambda|`` over the terms that affect each log sum.

    ``terms`` are ``log P + log lambda`` along the last axis and ``total`` their
    ``logsumexp``. Terms more than ``_NEGLIGIBLE_LOG_TERM`` below the total, and
    states with no predictive mass, do not affect the sum and are left out. The
    rest lie in ``[total - 40, total]``, so ``|term| <= |total| + 40``, and
    ``|log lambda| <= |term| + |log P|``.
    """
    finite_total = np.isfinite(total)
    significant = (
        terms >= np.where(finite_total, total - _NEGLIGIBLE_LOG_TERM, np.inf)[..., np.newaxis]
    )
    largest_log_state = np.max(
        np.broadcast_to(np.abs(log_state), terms.shape),
        axis=-1,
        where=significant,
        initial=0.0,
    )
    magnitude: DistributionArray = (
        np.where(finite_total, np.abs(total), 0.0)
        + _NEGLIGIBLE_LOG_TERM
        + 2 * largest_log_state
    )
    return magnitude


def _check_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        msg = f"{name} must be a positive integer; got {value!r}"
        raise ValueError(msg)


def monte_carlo_mark_pvalue(
    state_dist: ArrayLike,
    model: MarkModel,
    observed_marks: ArrayLike,
    *,
    n_samples: int = 1000,
    rng: np.random.Generator | int | None = None,
    return_samples: bool = False,
    batch_size: int = DEFAULT_MONTE_CARLO_BATCH_SIZE,
) -> MarkPredictiveCheck:
    """Monte Carlo predictive p-value of each event's observed mark.

    The rank-based predictive p-value of the paper, for marks that cannot be
    enumerated. The predictive mark density of an event is

    ``f_pred(y) = sum_x lambda(x, y) P(x) / sum_x Lambda(x) P(x)``,

    and the p-value is the probability that a mark ``Y`` drawn from ``f_pred``
    is no more probable than the observed mark:
    ``p = Pr[f_pred(Y) <= f_pred(y_obs)]``. It is estimated from ``n_samples``
    replicated marks per event: each draws a state from the event-weighted
    predictive distribution (:func:`~statespacecheck.event_weighted_predictive`),
    then a mark at that state with ``model.sample``. The comparison is made on
    log densities with a tolerance for their rounding error,
    ``16 * eps * (n_bins + M)``, where ``M`` bounds the magnitudes of the log
    terms that affect each sum (``log P``, ``log lambda`` and ``log Lambda``,
    before they cancel), so marks with equal predictive density count as ties
    at any scale of the inputs (compare the tie tolerance of
    :func:`~statespacecheck.mark_predictive_pvalue`). Small values mean the
    observed mark was unexpected given the prediction; an impossible mark
    gives 0.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_events, ...)
        Predictive state distribution for each event, where ``...`` represents
        one or more spatial axes. Rows need not be normalized.
    model : MarkModel
        The observation model: the log joint intensity ``log lambda(x, y)``
        (called with marks of shape ``(n, *mark_shape)``, it returns shape
        ``(n, ...)``, with ``-inf`` where the intensity is zero), a sampler of
        marks at flat state-bin indices, and the ground intensity, shape
        ``(...)``. Its parts must describe the same model; see
        :class:`MarkModel`.
    observed_marks : np.ndarray, shape (n_events, *mark_shape)
        The mark of each event.
    n_samples : int, optional
        Replicated marks per event. Default is 1000; the p-value's Monte Carlo
        standard error is ``sqrt(p (1 - p) / n_samples)``.
    rng : np.random.Generator, int, or None, optional
        Random number generator or seed. Default is None (fresh entropy).
    return_samples : bool, optional
        Also return the log predictive density of every replicated mark.
        Default is False.
    batch_size : int, optional
        Events processed at a time. Peak memory is about
        ``6 * batch_size * n_samples * n_bins * 8`` bytes (about 200 MB at the
        default 8 with 1000 samples and 512 bins); lower it for larger grids or
        more samples.

    Returns
    -------
    MarkPredictiveCheck
        The p-values, the log predictive density of each observed mark, and
        the replicated log densities if requested.

    Raises
    ------
    TypeError
        If ``model`` is not a :class:`MarkModel`.
    ValueError
        If shapes are inconsistent, inputs are negative or non-finite,
        ``n_samples`` or ``batch_size`` is not a positive integer, a callable
        returns output of the wrong shape (or NaN or ``+inf`` log
        intensities), an event's total predictive event intensity is zero, or
        a replicated mark has zero intensity at every state with predictive
        mass (the sampler and the intensity disagree, or the intensity
        underflowed before its log was taken).

    See Also
    --------
    mark_predictive_pvalue : The exact p-value for a finite set of marks.
    predictive_pvalue : A Monte Carlo p-value from a user-supplied sampler of
        whole time bins.

    Notes
    -----
    Results are reproducible: the same integer seed and the same
    ``batch_size`` give identical p-values. Changing ``batch_size`` changes
    which random numbers each event receives, so p-values then differ within
    Monte Carlo error.

    Examples
    --------
    Two marks, whose exact p-values :func:`~statespacecheck.mark_predictive_pvalue`
    also gives:

    >>> import numpy as np
    >>> from statespacecheck import MarkModel, mark_predictive_pvalue, monte_carlo_mark_pvalue
    >>> state = np.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    >>> rates = np.array([[4.0, 1.0], [1.0, 1.0], [1.0, 4.0]])  # (n_bins, n_marks)
    >>> def log_mark_intensity(marks):
    ...     return np.log(rates[:, marks].T)
    >>> def sample_marks(bins, rng):  # mark 1 with probability rates[x, 1] / Lambda(x)
    ...     return (rng.random(len(bins)) < rates[bins, 1] / rates[bins].sum(axis=1)).astype(
    ...         int
    ...     )
    >>> model = MarkModel(log_mark_intensity, sample_marks, rates.sum(axis=1))
    >>> check = monte_carlo_mark_pvalue(state, model, np.array([0, 1]), n_samples=2000, rng=0)
    >>> check.pvalue.round(1)
    array([1. , 0.3])
    >>> mark_predictive_pvalue(state, rates, np.array([0, 1])).round(3)
    array([1.   , 0.317])
    """
    # Checked for untyped callers (for example, a function passed where the model goes)
    supplied: object = model
    if not isinstance(supplied, MarkModel):
        msg = (
            "model must be a MarkModel(log_intensity, sample, ground_intensity); "
            f"got {type(model).__name__}"
        )
        raise TypeError(msg)
    _check_positive_integer(n_samples, "n_samples")
    _check_positive_integer(batch_size, "batch_size")
    state = _validate_state_distribution(state_dist, "state_dist")
    spatial_shape = np.shape(state_dist)[1:]
    ground = _validate_ground_intensity(model.ground_intensity, spatial_shape)
    marks = np.asarray(observed_marks)
    n_events = state.shape[0]
    if marks.ndim == 0 or marks.shape[0] != n_events:
        msg = (
            f"observed_marks must have one entry per event ({n_events}); "
            f"got shape {marks.shape}"
        )
        raise ValueError(msg)
    if np.issubdtype(marks.dtype, np.number):
        per_event = marks.reshape(n_events, int(np.prod(marks.shape[1:])))
        not_finite = np.flatnonzero(~np.isfinite(per_event).all(axis=1))
        if not_finite.size:
            msg = f"observed_marks must be finite; events {not_finite[:10].tolist()} are not"
            raise ValueError(msg)

    generator = np.random.default_rng(rng)
    pvalue: DistributionArray = np.empty(n_events)
    observed_log: DistributionArray = np.empty(n_events)
    simulated_log = np.empty((n_events, n_samples)) if return_samples else None
    for start in range(0, n_events, batch_size):
        stop = min(start + batch_size, n_events)
        batch_pvalue, batch_observed, batch_simulated = _monte_carlo_batch(
            state[start:stop],
            ground,
            marks[start:stop],
            model.log_intensity,
            model.sample,
            spatial_shape,
            n_samples,
            generator,
            start,
        )
        pvalue[start:stop] = batch_pvalue
        observed_log[start:stop] = batch_observed
        if simulated_log is not None:
            simulated_log[start:stop] = batch_simulated
    return MarkPredictiveCheck(pvalue, observed_log, simulated_log)
