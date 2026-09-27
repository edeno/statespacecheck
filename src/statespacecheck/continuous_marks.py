"""Per-event diagnostics of events whose marks are continuous or intractable.

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
with 32 waveform features can be ``exp(-140)``, below the smallest float32, in
which some decoders compute, and more features or distant marks go below the
smallest float64 (about ``exp(-745)``).
:func:`clusterless_event_diagnostics` computes all three per-event diagnostics
(HPD overlap, KL divergence and this p-value) from the same :class:`MarkModel`.

State-bin indices passed to a :data:`MarkSampler` are flat indices into the
state grid, in the C order of its spatial axes (as ``reshape(n, -1)`` flattens
them).
"""

from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ._validation import DistributionArray, as_array, flatten_time_spatial, validate_coverage
from .events import (
    EventDiagnostics,
    _check_predictive_rows,
    _check_time_bins,
    _first,
    _no_event_intensity_message,
    _normalize_log,
    _validate_ground_intensity,
    _validate_predictive,
    _validate_state_distribution,
    _validate_time_indices,
)
from .highest_density import DEFAULT_COVERAGE
from .state_consistency import (
    _log_space_kl,
    _underflowed_likelihood_rows,
    hpd_overlap,
    kl_divergence,
)

LogMarkIntensity: TypeAlias = Callable[[NDArray[Any]], NDArray[np.floating]]
"""Log of the joint intensity of marks at every state.

Called with marks of shape ``(n, *mark_shape)``; returns ``log lambda(x, y)`` for
each mark at every state bin as float64, shape ``(n, *spatial_shape)``: finite, or
``-inf`` where the intensity is zero. Compute it in log space (for example with
``scipy.stats.norm.logpdf``); exponentiating first can underflow. Observed marks
are passed read-only; copy them before modifying them.
"""

MarkSampler: TypeAlias = Callable[[NDArray[np.intp], np.random.Generator], NDArray[Any]]
"""Draws one mark for an event at each given state bin.

Called with flat state-bin indices of shape ``(n,)`` and a random number
generator; returns marks of shape ``(n, *mark_shape)``, each drawn from
``lambda(x, y) / Lambda(x)`` at its bin. It must draw only from the generator it
is given, so that seeded results are reproducible. The indices are read-only.
"""


class MarkModel(NamedTuple):
    """A marked point-process observation model, as the continuous-mark functions take it.

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


# Events processed per batch in :func:`monte_carlo_mark_pvalue` and
# :func:`clusterless_event_diagnostics`. Memory is
# dominated by arrays over every replicated mark at every state, batch x
# n_samples x n_bins x 8 B (8 x 1000 x 512 x 8 B ~ 33 MB); the peak is about two
# of them plus what the model's log_intensity allocates: ~90 MB for a model that
# allocates only its output, ~200 MB for a scipy.stats logpdf model. Larger batches
# are not faster.
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


def _evaluate_log_intensity(
    log_intensity: LogMarkIntensity,
    marks: NDArray[Any],
    spatial_shape: tuple[int, ...],
    first_event: int,
    replicates_per_event: int | None = None,
) -> DistributionArray:
    """Evaluate ``log_intensity`` at marks, checked, as a float64 ``(n, n_bins)`` array.

    ``marks`` are the observed marks of consecutive events from ``first_event``
    on or, with ``replicates_per_event``, that many replicated marks for each.
    Errors name the events. The result may share memory with the callable's
    output; do not modify it.
    """
    n = marks.shape[0]
    what = (
        "the observed marks" if replicates_per_event is None else "marks drawn by model.sample"
    )
    marks_per_event = 1 if replicates_per_event is None else replicates_per_event
    returned = as_array(
        log_intensity(marks),
        "model.log_intensity's output",
        "Return an ndarray with -inf where the intensity is zero",
    )
    dtype = returned.dtype
    if dtype.kind != "f":
        msg = (
            "model.log_intensity must return real floating-point log intensities, with "
            f"-inf for zero intensity; it returned {dtype}"
        )
        raise ValueError(msg)
    if dtype.itemsize < 8:
        msg = (
            f"model.log_intensity must return float64 values; it returned {dtype}, whose "
            "rounding (about 1e-7 for float32) splits marks of equal predictive density"
        )
        raise ValueError(msg)
    values = returned.astype(np.float64, copy=False)
    if values.shape != (n, *spatial_shape):
        msg = (
            f"model.log_intensity must return shape {(n, *spatial_shape)}, the log "
            f"intensity of each of the {n} marks at every state bin; got {values.shape}"
        )
        raise ValueError(msg)
    values = values.reshape(n, -1)
    largest = values.max(initial=-np.inf)  # NaN if any value is NaN
    if np.isnan(largest) or largest == np.inf:
        rows = np.flatnonzero(np.isnan(values).any(axis=1) | (values == np.inf).any(axis=1))
        events = np.unique(first_event + rows // marks_per_event)
        msg = (
            "model.log_intensity must return finite values, or -inf for zero intensity; "
            f"it returned NaN or +inf for {what} of events {_first(events)}"
        )
        raise ValueError(msg)
    return values


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
    observed_log_intensity: DistributionArray,
    model: MarkModel,
    *,
    mark_shape: tuple[int, ...],
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
    observed_log_intensity : np.ndarray, shape (n_batch, n_bins)
        Log intensity of each observed mark at every state bin, from
        :func:`_evaluate_log_intensity`.
    model : MarkModel
        The model; its ground intensity is ``ground``, already validated.
    mark_shape : tuple of int
        Shape of one mark, which ``model.sample`` must return per state bin.
    spatial_shape : tuple of int
        Shape of the state grid, which ``model.log_intensity`` must return per mark.
    n_samples : int
        Replicated marks per event.
    rng : np.random.Generator
        Used for the state draws, then by ``model.sample``.
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
    _check_zero_ground(np.isfinite(observed_log_intensity) & no_events, first_event)
    # The state's normalization cancels in the density ratio, so it is left out:
    # summing the row could overflow
    log_state = _safe_log(state)
    # log sum_x Lambda(x) P(x), the normalizer of the predictive mark density, and
    # the event-weighted distribution the replicates' states are drawn from
    norm_terms = log_state + _safe_log(ground)
    log_norm = logsumexp(norm_terms, axis=1)
    no_intensity = np.flatnonzero(np.isneginf(log_norm))
    if no_intensity.size:
        raise ValueError(_no_event_intensity_message(first_event + no_intensity))
    event_weighted = np.exp(norm_terms - log_norm[:, np.newaxis])

    observed_terms = log_state + observed_log_intensity
    observed_sum = logsumexp(observed_terms, axis=1)

    state_bins = _sample_state_bins(event_weighted, n_samples, rng)
    # Read-only: a sampler that reordered them in place would move replicates
    # between events without any check noticing
    flat_bins = state_bins.ravel()
    flat_bins.flags.writeable = False
    replicated_marks = as_array(
        model.sample(flat_bins, rng), "model.sample's output", "Return an ndarray of marks"
    )
    if replicated_marks.shape != (n_batch * n_samples, *mark_shape):
        msg = (
            f"model.sample must return marks of shape {(n_batch * n_samples, *mark_shape)}, "
            f"one per state bin, like the observed marks; got {replicated_marks.shape}"
        )
        raise ValueError(msg)
    not_finite = _nonfinite_rows(replicated_marks)
    if not_finite.size:
        events = np.unique(first_event + not_finite // n_samples)
        msg = f"model.sample returned non-finite marks for events {_first(events)}"
        raise ValueError(msg)
    replicated_log_intensity = _evaluate_log_intensity(
        model.log_intensity, replicated_marks, spatial_shape, first_event, n_samples
    ).reshape(n_batch, n_samples, n_bins)
    # A mark drawn at a state has positive intensity there. Zero intensity at its own
    # state means the sampler and the intensity disagree (for example, about the
    # order of the flat state-bin indices) or the intensity underflowed; the
    # replicate would then count toward the p-value with the wrong density
    at_drawn_state = np.take_along_axis(
        replicated_log_intensity, state_bins[:, :, np.newaxis], axis=2
    )[:, :, 0]
    impossible = np.isneginf(at_drawn_state)
    if impossible.any():
        events = first_event + np.flatnonzero(impossible.any(axis=1))
        msg = (
            f"{int(impossible.sum())} of {impossible.size} marks drawn by model.sample have "
            "zero intensity (model.log_intensity -inf) at the state bin they were drawn "
            f"at (events {_first(events)}). model.sample must draw from the model "
            "model.log_intensity describes, at flat state-bin indices in C order, and "
            "model.log_intensity must be computed in log space"
        )
        raise ValueError(msg)
    if no_events.any():
        _check_zero_ground(
            (np.isfinite(replicated_log_intensity) & no_events[:, np.newaxis, :]).any(axis=1),
            first_event,
        )
    # The (n_batch, n_samples, n_bins) arrays dominate memory: logsumexp copies its
    # input several times, so it is applied one event at a time
    log_joint = replicated_log_intensity + log_state[:, np.newaxis, :]
    simulated_sum = np.empty((n_batch, n_samples))
    for row in range(n_batch):
        simulated_sum[row] = logsumexp(log_joint[row], axis=1)

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
    """Raise if a log intensity is finite at a state with no ground intensity.

    ``finite_at_zero_ground`` has shape ``(n_batch, n_bins)``; the caller chooses
    which states it covers.
    """
    events = np.flatnonzero(finite_at_zero_ground.any(axis=1))
    if events.size:
        msg = (
            "model.log_intensity is finite at state bins where model.ground_intensity is "
            f"zero (events {_first(first_event + events)}); the ground intensity "
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


def _check_mark_model(model: object) -> None:
    """Raise ``TypeError`` unless ``model`` is a :class:`MarkModel`.

    Checked for untyped callers (for example, a function passed where the model goes).
    """
    if not isinstance(model, MarkModel):
        msg = (
            "model must be a MarkModel(log_intensity, sample, ground_intensity); "
            f"got {type(model).__name__}"
        )
        raise TypeError(msg)


def _nonfinite_rows(marks: NDArray[Any]) -> NDArray[np.intp]:
    """Return the indices of the numeric marks ``(n, *mark_shape)`` that are not all finite."""
    if not np.issubdtype(marks.dtype, np.number):
        return np.empty(0, dtype=np.intp)
    return np.flatnonzero(~np.isfinite(marks).all(axis=tuple(range(1, marks.ndim))))


def _validate_observed_marks(marks: ArrayLike, n_events: int, name: str) -> NDArray[Any]:
    """Check the marks of the events, returned read-only so the model cannot change them."""
    marks = as_array(marks, name, "Pass an ndarray of the events' marks")
    if marks.ndim == 0 or marks.shape[0] != n_events:
        msg = f"{name} must have one entry per event ({n_events}); got shape {marks.shape}"
        raise ValueError(msg)
    not_finite = _nonfinite_rows(marks)
    if not_finite.size:
        msg = f"{name} must be finite; events {_first(not_finite)} are not"
        raise ValueError(msg)
    read_only = marks.view()
    read_only.flags.writeable = False
    return read_only


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
        ``2 * batch_size * n_samples * n_bins * 8`` bytes plus what
        ``model.log_intensity`` allocates (at the default 8 with 1000 samples and
        512 bins, about 90 MB for a model that allocates only its output, and
        about twice that for one built on ``scipy.stats`` log densities); lower
        it for larger grids or more samples.

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
        If shapes are inconsistent, inputs are negative, non-finite or masked,
        ``n_samples`` or ``batch_size`` is not a positive integer, or an
        event's total predictive event intensity is zero. Also if a callable
        returns invalid output: log intensities of the wrong shape, not real
        floating point of at least float64 precision, or NaN or ``+inf``;
        replicated marks of the wrong shape, masked or non-finite; a
        finite log intensity where the ground intensity is zero and the state
        has mass; or a replicated mark with zero intensity at the state it was
        drawn at (the sampler and the intensity disagree, or the intensity
        underflowed before its log was taken). Errors name the events.

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
    _check_mark_model(model)
    _check_positive_integer(n_samples, "n_samples")
    _check_positive_integer(batch_size, "batch_size")
    state = _validate_state_distribution(state_dist, "state_dist")
    spatial_shape = np.shape(state_dist)[1:]
    ground = _validate_ground_intensity(model.ground_intensity, spatial_shape)
    n_events = state.shape[0]
    marks = _validate_observed_marks(observed_marks, n_events, "observed_marks")

    generator = np.random.default_rng(rng)
    pvalue: DistributionArray = np.empty(n_events)
    observed_log: DistributionArray = np.empty(n_events)
    simulated_log = np.empty((n_events, n_samples)) if return_samples else None
    for start in range(0, n_events, batch_size):
        stop = min(start + batch_size, n_events)
        batch_pvalue, batch_observed, batch_simulated = _monte_carlo_batch(
            state[start:stop],
            ground,
            _evaluate_log_intensity(
                model.log_intensity, marks[start:stop], spatial_shape, start
            ),
            model,
            mark_shape=marks.shape[1:],
            spatial_shape=spatial_shape,
            n_samples=n_samples,
            rng=generator,
            first_event=start,
        )
        pvalue[start:stop] = batch_pvalue
        observed_log[start:stop] = batch_observed
        if simulated_log is not None:
            simulated_log[start:stop] = batch_simulated
    return MarkPredictiveCheck(pvalue, observed_log, simulated_log)


def clusterless_event_diagnostics(
    predictive: ArrayLike,
    model: MarkModel,
    event_time_ind: ArrayLike,
    event_marks: ArrayLike,
    *,
    coverage: float = DEFAULT_COVERAGE,
    n_samples: int = 1000,
    rng: np.random.Generator | int | None = None,
    return_likelihood: bool = False,
    batch_size: int = DEFAULT_MONTE_CARLO_BATCH_SIZE,
) -> EventDiagnostics:
    """Compute HPD overlap, KL divergence, and predictive p-value for events with any marks.

    The per-event diagnostics of :func:`~statespacecheck.event_diagnostics` for
    marks that cannot be enumerated, such as the waveform features of
    clusterless decoding; they apply to any mark space the model can sample.
    Each event is compared with the one-step predictive distribution ``P`` of
    its time bin:

    - The single-event likelihood is the joint intensity of the observed mark
      normalized over states, ``Q(x) = lambda(x, y_obs) / sum_u lambda(u, y_obs)``
      (computed in log space); HPD overlap and KL divergence compare it with
      ``P``.
    - The predictive p-value is :func:`monte_carlo_mark_pvalue`'s, from the
      predictive mark density
      ``f_pred(y) = sum_x lambda(x, y) P(x) / sum_x Lambda(x) P(x)``.

    With a finite set of marks, such as the units of spike-sorted data, use
    :func:`~statespacecheck.event_diagnostics`: its p-value is exact. Written
    as a :class:`MarkModel` of integer marks whose ``log_intensity`` is
    ``np.log`` of the same (float64) intensity table, those marks give the same HPD
    overlap, KL divergence and likelihood bit for bit, and the same p-values
    within Monte Carlo error.

    Parameters
    ----------
    predictive : np.ndarray, shape (n_time, ...)
        One-step predictive state distribution ``p(x_t | y_{1:t-1})`` at each
        time bin, where ``...`` represents one or more spatial axes. Only the
        time bins that events use are checked and used.
    model : MarkModel
        The observation model: the log joint intensity ``log lambda(x, y)``
        (called with marks of shape ``(n, *mark_shape)``, it returns shape
        ``(n, ...)``, with ``-inf`` where the intensity is zero), a sampler of
        marks at flat state-bin indices, and the ground intensity, shape
        ``(...)``. Its parts must describe the same model; see
        :class:`MarkModel`.
    event_time_ind : np.ndarray, shape (n_events,)
        Time-bin index of each event. Events that share a time bin are each
        compared with that bin's predictive distribution.
    event_marks : np.ndarray, shape (n_events, *mark_shape)
        The mark of each event, for example its waveform features.
    coverage : float, default 0.95
        Coverage probability of the HPD regions.
    n_samples : int, default 1000
        Replicated marks per event for the p-value; its Monte Carlo standard
        error is ``sqrt(p (1 - p) / n_samples)``.
    rng : np.random.Generator, int, or None, optional
        Random number generator or seed. Default is None (fresh entropy).
    return_likelihood : bool, default False
        If True, also return each event's normalized likelihood,
        shape ``(n_events, ...)``.
    batch_size : int, default 8
        Events processed at a time; see :func:`monte_carlo_mark_pvalue` for its
        effect on memory.

    Returns
    -------
    EventDiagnostics
        Per-event ``hpd_overlap``, ``kl_divergence``, and ``predictive_pvalue``
        arrays of shape ``(n_events,)``, plus ``likelihood`` if requested.

    Raises
    ------
    TypeError
        If ``model`` is not a :class:`MarkModel`.
    ValueError
        If shapes are inconsistent, ``event_time_ind`` is not integer time-bin
        indices or is out of range, inputs are negative, non-finite or masked,
        ``n_samples`` or ``batch_size`` is not a
        positive integer, or ``coverage`` is outside ``(0, 1)``; if an event's
        time bin has no predictive mass where the ground intensity is
        positive, or its observed mark has zero intensity at every state or a
        finite log intensity where the ground intensity is zero; or if a
        callable returns invalid output (see :func:`monte_carlo_mark_pvalue`).
        Errors name events and time bins by their index in the inputs.

    See Also
    --------
    event_diagnostics : The exact diagnostics for a finite set of marks.
    monte_carlo_mark_pvalue : The p-value alone, for given state distributions.

    Notes
    -----
    The p-values are :func:`monte_carlo_mark_pvalue`'s for
    ``predictive[event_time_ind]``: the same seed and ``batch_size`` give
    identical results. HPD overlap, KL divergence and the likelihood do not
    depend on the seed. They depend on ``batch_size`` only in the last bit,
    and only if ``model.log_intensity`` computes a mark's values differently
    depending on how many marks it is called with (a matrix product can).

    The returned likelihood is exponentiated after normalizing in log space, so
    it is exactly 0 at states where its log is more than about 745 below the
    largest (a ratio below the smallest float64, about ``5e-324``). The KL
    divergence at such states is computed from the log intensity, so it is
    ``+inf`` only where the prediction has mass and the intensity is zero
    (disjoint supports).

    Examples
    --------
    Two units' place fields written as a model of integer marks, whose exact
    diagnostics :func:`~statespacecheck.event_diagnostics` also gives:

    >>> import numpy as np
    >>> from statespacecheck import MarkModel, clusterless_event_diagnostics, event_diagnostics
    >>> predictive = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])  # (n_time, n_bins)
    >>> place_fields = np.array([[5.0, 0.1], [1.0, 1.0], [0.1, 5.0]])  # (n_bins, n_marks)
    >>> def log_mark_intensity(marks):
    ...     return np.log(place_fields[:, marks].T)
    >>> def sample_marks(bins, rng):  # unit 1 with probability place_fields[x, 1] / Lambda(x)
    ...     unit_1 = place_fields[bins, 1] / place_fields[bins].sum(axis=1)
    ...     return (rng.random(len(bins)) < unit_1).astype(int)
    >>> model = MarkModel(log_mark_intensity, sample_marks, place_fields.sum(axis=1))
    >>> time_ind, marks = np.array([0, 1]), np.array([0, 0])
    >>> result = clusterless_event_diagnostics(
    ...     predictive, model, time_ind, marks, n_samples=5000, rng=0
    ... )
    >>> result.predictive_pvalue.round(2)
    array([1.  , 0.17])
    >>> exact = event_diagnostics(predictive, place_fields, time_ind, marks)
    >>> exact.predictive_pvalue.round(3)
    array([1.   , 0.172])
    >>> bool(np.array_equal(result.kl_divergence, exact.kl_divergence))
    True
    """
    _check_mark_model(model)
    validate_coverage(coverage)
    _check_positive_integer(n_samples, "n_samples")
    _check_positive_integer(batch_size, "batch_size")
    # Kept in its dtype and converted to float64 batch by batch: a decoder's
    # float32 predictive would double in size
    predictive = _validate_predictive(predictive)
    spatial_shape = predictive.shape[1:]
    ground = _validate_ground_intensity(model.ground_intensity, spatial_shape)
    time_ind = _validate_time_indices(event_time_ind, predictive.shape[0])
    n_events = time_ind.shape[0]
    marks = _validate_observed_marks(event_marks, n_events, "event_marks")
    predictive_flat = flatten_time_spatial(predictive)
    _check_predictive_rows(predictive_flat, time_ind)
    # An event needs a state with predictive mass where events occur (checked here,
    # before the model is called, rather than batch by batch)
    has_events = ground > 0.0

    def has_mass_where_events_occur(rows: NDArray[Any]) -> NDArray[np.bool_]:
        overlap: NDArray[np.bool_] = np.count_nonzero((rows > 0) & has_events, axis=1) > 0
        return overlap

    _check_time_bins(
        predictive_flat,
        time_ind,
        has_mass_where_events_occur,
        "At time bins {bins} the predictive distribution puts no probability where "
        "model.ground_intensity is positive, so the mark distribution is undefined; "
        "used by events {events}",
    )

    generator = np.random.default_rng(rng)
    event_hpd: DistributionArray = np.empty(n_events)
    event_kl: DistributionArray = np.empty(n_events)
    event_pvalue: DistributionArray = np.empty(n_events)
    likelihood: DistributionArray | None = (
        np.empty((n_events, ground.shape[0])) if return_likelihood else None
    )
    for start in range(0, n_events, batch_size):
        stop = min(start + batch_size, n_events)
        predictive_batch = np.asarray(predictive_flat[time_ind[start:stop]], dtype=np.float64)
        observed_log_intensity = _evaluate_log_intensity(
            model.log_intensity, marks[start:stop], spatial_shape, start
        )
        # The likelihood is normalized over every state, so the intensity must agree
        # with the ground intensity at states without predictive mass too
        _check_zero_ground(np.isfinite(observed_log_intensity) & (ground == 0.0), start)
        no_likelihood = np.flatnonzero(np.isneginf(observed_log_intensity).all(axis=1))
        if no_likelihood.size:
            msg = (
                f"The observed marks of events {_first(start + no_likelihood)} have zero "
                "intensity at every state (model.log_intensity is -inf everywhere), so "
                "they have no likelihood"
            )
            raise ValueError(msg)
        # C order, as event_likelihood receives it from event_diagnostics: the
        # normalizing sum's rounding depends on the layout
        likelihood_batch = _normalize_log(np.ascontiguousarray(observed_log_intensity))
        event_hpd[start:stop] = hpd_overlap(
            predictive_batch, likelihood_batch, coverage=coverage
        )
        event_kl[start:stop] = kl_divergence(predictive_batch, likelihood_batch)
        # Where the likelihood underflowed, its log gives the divergence
        rows = _underflowed_likelihood_rows(
            predictive_batch, likelihood_batch, np.isfinite(observed_log_intensity)
        )
        if rows.size:
            event_kl[start + rows] = _log_space_kl(
                predictive_batch[rows], observed_log_intensity[rows]
            )
        event_pvalue[start:stop], _, _ = _monte_carlo_batch(
            predictive_batch,
            ground,
            observed_log_intensity,
            model,
            mark_shape=marks.shape[1:],
            spatial_shape=spatial_shape,
            n_samples=n_samples,
            rng=generator,
            first_event=start,
        )
        if likelihood is not None:
            likelihood[start:stop] = likelihood_batch

    return EventDiagnostics(
        hpd_overlap=event_hpd,
        kl_divergence=event_kl,
        predictive_pvalue=event_pvalue,
        likelihood=None
        if likelihood is None
        else likelihood.reshape(n_events, *spatial_shape),
    )
