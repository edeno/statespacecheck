"""Per-event diagnostics for marked point-process observations.

Spike trains (and other event streams) are usually decoded with a marked
point-process observation model: each event carries a discrete mark (for
spike-sorted data, the identity of the unit that fired) and each mark has a
state-dependent intensity. The functions here turn such a model into the
per-event quantities that the distribution-level diagnostics in
:mod:`statespacecheck.state_consistency` compare:

- :func:`event_likelihood` normalizes one event's mark intensity over the state
  space, giving the single-event likelihood.
- :func:`predictive_mark_probabilities` gives the predictive probability of each
  mark for the next event.
- :func:`event_weighted_predictive` gives the state distribution of the next
  event, weighting the predictive distribution by the total event intensity.
- :func:`mark_predictive_pvalue` evaluates the predictive check exactly by
  summing over the finite set of marks.
- :func:`event_diagnostics` computes HPD overlap, KL divergence, and the exact
  predictive p-value for every event in a recording.
- :func:`baseline_threshold` estimates a flagging threshold from a baseline
  (well-specified) sample of per-event values.
- :func:`flag_events` applies the paper's flagging rule to every event.

Array conventions follow the rest of the package: state distributions are
``(n_events, ...)`` or ``(n_time, ...)`` where ``...`` is one or more spatial
axes, and mark intensity tables are ``(..., n_marks)`` over the same spatial
axes.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ._validation import (
    DistributionArray,
    check_threshold_not_nan,
    flatten_time_spatial,
    validate_coverage,
)
from .highest_density import DEFAULT_COVERAGE
from .state_consistency import hpd_overlap, kl_divergence

# Events processed per batch in :func:`event_diagnostics`. Bounds the
# (batch, n_bins) working arrays so recordings with ~10^6 events do not
# allocate multi-GB buffers: 50 000 events x 512 bins x 8 B ~ 200 MB each.
DEFAULT_EVENT_BATCH_SIZE = 50_000


class EventDiagnostics(NamedTuple):
    """Per-event diagnostic values returned by :func:`event_diagnostics`.

    Attributes
    ----------
    hpd_overlap : np.ndarray, shape (n_events,)
        HPD overlap between the predictive distribution and each event's
        likelihood. Low values indicate poor local fit.
    kl_divergence : np.ndarray, shape (n_events,)
        KL divergence from the predictive distribution to each event's
        likelihood. High values indicate poor local fit.
    predictive_pvalue : np.ndarray, shape (n_events,)
        Exact predictive p-value of each event's observed mark (see
        :func:`mark_predictive_pvalue`). Low values indicate poor local fit.
    likelihood : np.ndarray, shape (n_events, ...), or None
        Normalized single-event likelihood of each event over the state space.
        ``None`` unless ``return_likelihood=True``.
    """

    hpd_overlap: NDArray[np.float64]
    kl_divergence: NDArray[np.float64]
    predictive_pvalue: NDArray[np.float64]
    likelihood: NDArray[np.float64] | None


def _flatten_mark_intensities(
    mark_intensities: ArrayLike, spatial_shape: tuple[int, ...]
) -> DistributionArray:
    """Validate a ``(..., n_marks)`` table and flatten it to ``(n_bins, n_marks)``."""
    table = np.asarray(mark_intensities, dtype=np.float64)
    if table.shape[:-1] != spatial_shape or table.ndim < 2:
        msg = (
            f"mark_intensities must have shape {(*spatial_shape, 'n_marks')} to match the "
            f"state distribution's spatial axes; got {table.shape}"
        )
        if table.ndim >= 2 and table.shape[1:] == spatial_shape:
            fix = (
                "mark_intensities.T"
                if table.ndim == 2
                else "np.moveaxis(mark_intensities, 0, -1)"
            )
            msg += (
                f". It looks like (n_marks, ...), e.g. place fields stored one unit "
                f"per row; pass {fix}"
            )
        raise ValueError(msg)
    if table.shape[-1] == 0:
        msg = "mark_intensities must contain at least one mark"
        raise ValueError(msg)
    if not np.all(np.isfinite(table)) or np.any(table < 0.0):
        msg = "mark_intensities must contain only finite nonnegative values"
        raise ValueError(msg)
    return table.reshape(-1, table.shape[-1])


def _validate_state_distribution(state_dist: ArrayLike, name: str) -> DistributionArray:
    """Validate a ``(n_events, ...)`` distribution and flatten it to ``(n_events, n_bins)``."""
    state_dist = np.asarray(state_dist, dtype=float)
    if state_dist.ndim < 2:
        msg = (
            f"{name} must have shape (n_events, ...) with at least one spatial axis; "
            f"got shape {state_dist.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(state_dist)) or np.any(state_dist < 0.0):
        msg = f"{name} must contain only finite nonnegative values"
        raise ValueError(msg)
    return flatten_time_spatial(state_dist)


def _validate_marks(marks: ArrayLike, n_marks: int, name: str) -> NDArray[np.intp]:
    """Check that ``marks`` is a 1-D integer array of valid mark indices."""
    marks = np.asarray(marks)
    if marks.ndim == 1 and marks.size == 0:
        return np.empty(0, dtype=np.intp)
    if marks.ndim != 1 or not np.issubdtype(marks.dtype, np.integer):
        msg = (
            f"{name} must be a 1-D integer array; got shape {marks.shape}, dtype {marks.dtype}"
        )
        raise ValueError(msg)
    if marks.size and (marks.min() < 0 or marks.max() >= n_marks):
        msg = f"{name} must lie in [0, {n_marks}); got values outside that range"
        raise ValueError(msg)
    return marks.astype(np.intp, copy=False)


def _first(indices: NDArray[np.intp]) -> list[int]:
    """Return the first ten indices, for error messages."""
    return [int(i) for i in indices[:10]]


def _check_event_inputs(
    predictive_flat: DistributionArray,
    rates: NDArray[np.floating],
    time_ind: NDArray[np.intp],
    marks: NDArray[np.intp],
) -> None:
    """Check the time bins and marks the events use, reporting absolute indices.

    ``predictive_flat`` is ``(n_time, n_bins)`` and ``rates`` is
    ``(n_bins, n_marks)``. Only rows and marks referenced by an event are
    checked, as only those enter the diagnostics.
    """
    n_time = predictive_flat.shape[0]
    used_time = np.zeros(n_time, dtype=bool)
    used_time[time_ind] = True

    invalid_row = ~np.isfinite(predictive_flat).all(axis=1) | (predictive_flat < 0.0).any(
        axis=1
    )
    bad_time = np.flatnonzero(invalid_row & used_time)
    if bad_time.size:
        events = np.flatnonzero(np.isin(time_ind, bad_time))
        msg = (
            "predictive must contain only finite nonnegative values; "
            f"time bins {_first(bad_time)} do not (used by events {_first(events)})"
        )
        raise ValueError(msg)

    # Marks that events use and whose intensity is zero everywhere
    bad_marks = np.intersect1d(np.flatnonzero(~(rates > 0.0).any(axis=0)), marks)
    if bad_marks.size:
        events = np.flatnonzero(np.isin(marks, bad_marks))
        msg = (
            f"mark_intensities is zero everywhere for marks {_first(bad_marks)}, so "
            f"their events have no likelihood; used by events {_first(events)}"
        )
        raise ValueError(msg)

    # Expected total event intensity per time bin under the prediction.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        total = predictive_flat @ rates.sum(axis=1)
    no_events = used_time & ~(np.isfinite(total) & (total > 0.0))
    bad_time = np.flatnonzero(no_events)
    if bad_time.size:
        events = np.flatnonzero(np.isin(time_ind, bad_time))
        msg = (
            f"At time bins {_first(bad_time)} the predictive distribution puts no "
            "probability where any mark has intensity (or the total overflows), so "
            f"the mark distribution is undefined; used by events {_first(events)}"
        )
        raise ValueError(msg)


def event_likelihood(event_intensities: ArrayLike) -> DistributionArray:
    """Normalize event intensities over the state space.

    For a marked point-process observation model, the likelihood contribution
    of a single event with mark ``c`` is proportional to that mark's intensity
    ``lambda_c(x)`` (or expected count ``lambda_c(x) * dt``). This function
    normalizes each row to sum to 1 over the spatial axes, giving the
    single-event likelihood that :func:`~statespacecheck.hpd_overlap` and
    :func:`~statespacecheck.kl_divergence` compare with the predictive
    distribution.

    The Poisson exposure term ``exp(-sum_c lambda_c(x) * dt)`` is shared by
    every event in a time bin, so it is deliberately left out: attaching it to
    each event would count it once per event when a bin contains several. A
    common bin width ``dt`` cancels on normalization, so rates and expected
    counts give the same result.

    Normalization is done in log space, so rows with tiny but nonzero
    intensities keep their shape (``[1e-20, 2e-20, 4e-20]`` becomes
    ``[1/7, 2/7, 4/7]``).

    Parameters
    ----------
    event_intensities : np.ndarray, shape (n_events, ...)
        Nonnegative state-dependent intensity (or expected count) of each
        event's mark, where ``...`` represents one or more spatial axes.

    Returns
    -------
    likelihood : np.ndarray, shape (n_events, ...)
        Single-event likelihood; each row sums to 1 over the spatial axes.

    Raises
    ------
    ValueError
        If the input has no spatial axis, contains negative or non-finite
        values, or has a row that is zero everywhere (no defined likelihood).

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import event_likelihood
    >>> event_likelihood(np.array([[1.0, 2.0, 1.0]]))
    array([[0.25, 0.5 , 0.25]])
    """
    event_intensities = np.asarray(event_intensities, dtype=np.float64)
    if event_intensities.ndim < 2 or np.prod(event_intensities.shape[1:]) == 0:
        msg = (
            "event_intensities must have shape (n_events, ...) with a non-empty spatial "
            f"axis; got shape {event_intensities.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(event_intensities)) or np.any(event_intensities < 0.0):
        msg = "event_intensities must contain only finite nonnegative values"
        raise ValueError(msg)
    flat = flatten_time_spatial(event_intensities)
    with np.errstate(divide="ignore"):
        log_intensity = np.log(flat)
    log_norm = logsumexp(log_intensity, axis=-1, keepdims=True)
    degenerate = np.isneginf(log_norm[:, 0])
    if np.any(degenerate):
        bad = np.flatnonzero(degenerate)
        msg = (
            "Cannot compute an event likelihood for rows that are zero everywhere; "
            f"row indices: {_first(bad)}"
        )
        raise ValueError(msg)
    likelihood: DistributionArray = np.exp(log_intensity - log_norm)
    return likelihood.reshape(event_intensities.shape)


def predictive_mark_probabilities(
    state_dist: ArrayLike, mark_intensities: ArrayLike
) -> DistributionArray:
    """Compute the predictive probability of each mark for the next event.

    Mark intensities are averaged over the state distribution and then
    normalized across marks:

    ``q[c] = sum_x p[x] * lambda_c(x) / sum_d sum_x p[x] * lambda_d(x)``.

    This is the mark distribution of a randomly selected event under the
    predictive distribution. Normalizing across marks at each state before
    averaging would drop the weighting by the state-dependent total event
    rate, and is only equivalent when that total is constant across states.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_events, ...)
        Predictive state distribution for each event, where ``...`` represents
        one or more spatial axes.
    mark_intensities : np.ndarray, shape (..., n_marks)
        Nonnegative intensity (or expected count) of every mark at every state.

    Returns
    -------
    mark_probabilities : np.ndarray, shape (n_events, n_marks)
        Predictive mark probabilities; each row sums to 1.

    Raises
    ------
    ValueError
        If shapes are inconsistent, inputs are negative or non-finite, or a
        row has zero (or non-finite) total predictive event intensity, for
        which the mark distribution is undefined.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import predictive_mark_probabilities
    >>> state = np.array([[0.5, 0.5]])
    >>> intensities = np.array([[1.0, 0.0], [1.0, 2.0]])  # (n_bins, n_marks)
    >>> predictive_mark_probabilities(state, intensities)
    array([[0.5, 0.5]])
    """
    state = _validate_state_distribution(state_dist, "state_dist")
    rates = _flatten_mark_intensities(mark_intensities, np.shape(state_dist)[1:])

    # Finite inputs can still overflow in the product or the sum across marks;
    # report that as a contract error rather than dividing by infinity. The
    # product never divides, but NumPy < 2.3 on macOS (Accelerate) raises a
    # spurious divide-by-zero flag here, so that flag is ignored too.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expected_intensities: NDArray[np.floating] = state @ rates
        total_intensity = expected_intensities.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(expected_intensities)):
        msg = "Predictive expected mark intensities are non-finite after integration"
        raise ValueError(msg)
    nonfinite_total = ~np.isfinite(total_intensity[:, 0])
    if nonfinite_total.any():
        bad = np.flatnonzero(nonfinite_total)
        msg = f"Predictive total event intensity is non-finite for row indices: {_first(bad)}"
        raise ValueError(msg)
    zero_total = total_intensity[:, 0] == 0.0
    if zero_total.any():
        bad = np.flatnonzero(zero_total)
        msg = (
            "Predictive mark distribution is undefined for rows with zero total "
            f"event intensity; row indices: {_first(bad)}"
        )
        raise ValueError(msg)
    mark_probabilities: DistributionArray = expected_intensities / total_intensity
    return mark_probabilities


def _validate_ground_intensity(
    ground_intensity: ArrayLike, spatial_shape: tuple[int, ...]
) -> DistributionArray:
    """Check the ground intensity against the state grid and flatten it to ``(n_bins,)``."""
    ground = np.asarray(ground_intensity, dtype=np.float64)
    if ground.shape != spatial_shape:
        msg = (
            f"ground_intensity must have shape {spatial_shape} to match the state "
            f"distribution's spatial axes; got {ground.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(ground)) or np.any(ground < 0.0):
        msg = "ground_intensity must contain only finite nonnegative values"
        raise ValueError(msg)
    return ground.ravel()


def event_weighted_predictive(
    state_dist: ArrayLike, ground_intensity: ArrayLike
) -> DistributionArray:
    """Compute the state distribution of the next event.

    ``P_event(x) = Lambda(x) P(x) / sum_u Lambda(u) P(u)``, where ``P`` is the
    predictive state distribution and ``Lambda`` the ground intensity, the
    total event intensity at each state. A randomly chosen event is more
    likely to come from states with a higher total event intensity, so the
    state of an event is distributed as ``P`` weighted by ``Lambda``. The two
    are equal when the ground intensity is constant.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_events, ...)
        Predictive state distribution for each event, where ``...`` represents
        one or more spatial axes. Rows need not be normalized.
    ground_intensity : np.ndarray, shape (...)
        Nonnegative total event intensity at every state, over the same
        spatial axes. For sorted marks with intensities ``mark_intensities``
        of shape ``(..., n_marks)``, this is ``mark_intensities.sum(axis=-1)``.

    Returns
    -------
    event_weighted : np.ndarray, shape (n_events, ...)
        Event-weighted state distribution; each row sums to 1.

    Raises
    ------
    ValueError
        If shapes are inconsistent, inputs are negative or non-finite, or a
        row has zero (or non-finite) total event intensity under the state
        distribution, for which the event's state is undefined.

    See Also
    --------
    predictive_mark_probabilities : The mark distribution of the next event.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import event_weighted_predictive
    >>> state = np.array([[0.5, 0.5]])
    >>> event_weighted_predictive(state, np.array([1.0, 3.0]))
    array([[0.25, 0.75]])
    """
    state = _validate_state_distribution(state_dist, "state_dist")
    ground = _validate_ground_intensity(ground_intensity, np.shape(state_dist)[1:])
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = state * ground
        total = weighted.sum(axis=1, keepdims=True)
    undefined = ~np.isfinite(total[:, 0]) | (total[:, 0] == 0.0)
    if undefined.any():
        msg = (
            "Event-weighted predictive distribution is undefined for rows with zero or "
            f"non-finite total event intensity; row indices: {_first(np.flatnonzero(undefined))}"
        )
        raise ValueError(msg)
    event_weighted: DistributionArray = (weighted / total).reshape(np.shape(state_dist))
    return event_weighted


def mark_predictive_pvalue(
    state_dist: ArrayLike,
    mark_intensities: ArrayLike,
    observed_marks: ArrayLike,
) -> DistributionArray:
    """Exact predictive p-value of each event's observed mark.

    With a finite set of marks, the predictive check can be evaluated exactly
    rather than by Monte Carlo (compare :func:`~statespacecheck.predictive_pvalue`).
    For each event, the p-value is the probability that a mark drawn from the
    predictive mark distribution ``q`` (:func:`predictive_mark_probabilities`)
    is no more probable than the observed mark:

    ``p = sum_c q[c] * 1{q[c] <= q[observed]}``.

    Small values mean the observed mark was unexpected given the predictive
    state distribution. A small absolute tolerance on the ``<=`` comparison,
    ``16 * eps * n_bins`` times the event's largest predictive mark
    probability, absorbs floating-point reduction-order noise, so marks with
    equal predictive probability receive equal p-values across platforms.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_events, ...)
        Predictive state distribution for each event, where ``...`` represents
        one or more spatial axes.
    mark_intensities : np.ndarray, shape (..., n_marks)
        Nonnegative intensity (or expected count) of every mark at every state.
    observed_marks : np.ndarray, shape (n_events,)
        Integer index of the observed mark of each event.

    Returns
    -------
    pvalue : np.ndarray, shape (n_events,)
        Exact predictive p-values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If shapes are inconsistent, marks are out of range, or the predictive
        mark distribution is undefined (see :func:`predictive_mark_probabilities`).

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import mark_predictive_pvalue
    >>> state = np.array([[1.0, 0.0], [1.0, 0.0]])
    >>> intensities = np.array([[8.0, 1.0, 1.0], [1.0, 1.0, 8.0]])  # (n_bins, n_marks)
    >>> mark_predictive_pvalue(state, intensities, np.array([0, 2]))
    array([1. , 0.2])
    """
    mark_probabilities = predictive_mark_probabilities(state_dist, mark_intensities)
    n_events, n_marks = mark_probabilities.shape
    marks = _validate_marks(observed_marks, n_marks, "observed_marks")
    if marks.shape[0] != n_events:
        msg = (
            f"observed_marks must have one entry per event ({n_events}); got {marks.shape[0]}"
        )
        raise ValueError(msg)
    n_bins = int(np.prod(np.shape(state_dist)[1:]))
    observed = mark_probabilities[np.arange(n_events), marks]
    # Scaled by each event's own largest probability, so other events cannot change it
    relative_tolerance = float(np.finfo(mark_probabilities.dtype).eps * n_bins * 16)
    atol = relative_tolerance * mark_probabilities.max(axis=1)
    no_more_probable = mark_probabilities <= (observed + atol)[:, None]
    pvalue: DistributionArray = (mark_probabilities * no_more_probable).sum(axis=1)
    # The sum can exceed one by a few ulps; clip only that representational error.
    np.minimum(pvalue, 1.0, out=pvalue)
    return pvalue


def event_diagnostics(
    predictive: ArrayLike,
    mark_intensities: ArrayLike,
    event_time_ind: ArrayLike,
    event_marks: ArrayLike,
    *,
    coverage: float = DEFAULT_COVERAGE,
    return_likelihood: bool = False,
    batch_size: int = DEFAULT_EVENT_BATCH_SIZE,
) -> EventDiagnostics:
    """Compute HPD overlap, KL divergence, and predictive p-value for every event.

    Each event is compared with the one-step predictive distribution of its
    time bin. The event's likelihood is its mark intensity normalized over the
    state space (:func:`event_likelihood`), and the predictive p-value is the
    exact finite-mark check (:func:`mark_predictive_pvalue`). Events are
    processed in batches to bound memory for long recordings.

    Parameters
    ----------
    predictive : np.ndarray, shape (n_time, ...)
        One-step predictive state distribution ``p(x_t | y_{1:t-1})`` at each
        time bin, where ``...`` represents one or more spatial axes. Any other
        state distribution (for example a smoother) can be substituted.
    mark_intensities : np.ndarray, shape (..., n_marks)
        Nonnegative intensity (or expected count per bin) of every mark at every
        state, for example each unit's place field.
    event_time_ind : np.ndarray, shape (n_events,)
        Time-bin index of each event. If several events share a time bin, list
        each one separately; all are compared with that bin's predictive
        distribution, so events of the same mark in the same bin receive
        identical diagnostics.
    event_marks : np.ndarray, shape (n_events,)
        Mark index of each event (for spike-sorted data, the unit that fired).
    coverage : float, default 0.95
        Coverage probability of the HPD regions.
    return_likelihood : bool, default False
        If True, also return each event's normalized likelihood,
        shape ``(n_events, ...)``.
    batch_size : int, default 50_000
        Number of events processed at once.

    Returns
    -------
    EventDiagnostics
        Per-event ``hpd_overlap``, ``kl_divergence``, and ``predictive_pvalue``
        arrays of shape ``(n_events,)``, plus ``likelihood`` if requested.

    Raises
    ------
    ValueError
        If shapes are inconsistent, indices are out of range, or inputs are
        negative or non-finite; if an event's mark has zero intensity at every
        position; or if the predictive distribution of an event's time bin
        puts no mass where any mark has intensity (or the total overflows).

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import event_diagnostics
    >>> predictive = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])  # (n_time, n_bins)
    >>> place_fields = np.array([[5.0, 0.1], [1.0, 1.0], [0.1, 5.0]])  # (n_bins, n_marks)
    >>> result = event_diagnostics(
    ...     predictive, place_fields, np.array([0, 1]), np.array([0, 0])
    ... )
    >>> result.predictive_pvalue.round(3)
    array([1.   , 0.172])
    """
    validate_coverage(coverage)
    if batch_size < 1:
        msg = f"batch_size must be at least 1; got {batch_size}"
        raise ValueError(msg)
    predictive = np.asarray(predictive)
    if predictive.ndim < 2:
        msg = (
            "predictive must have shape (n_time, ...) with at least one spatial axis; "
            f"got shape {predictive.shape}"
        )
        raise ValueError(msg)
    spatial_shape = predictive.shape[1:]
    rates = _flatten_mark_intensities(mark_intensities, spatial_shape)
    n_time = predictive.shape[0]
    # An empty list is a float array too; empty event lists are accepted
    time_values = np.asarray(event_time_ind)
    if time_values.size and np.issubdtype(time_values.dtype, np.floating):
        msg = (
            "event_time_ind must be a 1-D integer array of time-bin indices, not times: "
            "convert event times with, e.g., np.digitize(event_times, time_bin_edges) - 1"
        )
        raise ValueError(msg)
    time_ind = _validate_marks(event_time_ind, n_time, "event_time_ind")
    marks = _validate_marks(event_marks, rates.shape[1], "event_marks")
    if time_ind.shape != marks.shape:
        msg = (
            "event_time_ind and event_marks must have the same length; got "
            f"{time_ind.shape[0]} and {marks.shape[0]}"
        )
        raise ValueError(msg)
    predictive_flat = flatten_time_spatial(predictive)
    _check_event_inputs(predictive_flat, rates, time_ind, marks)

    n_events = time_ind.shape[0]
    event_hpd: DistributionArray = np.empty(n_events)
    event_kl: DistributionArray = np.empty(n_events)
    event_pvalue: DistributionArray = np.empty(n_events)
    likelihood: DistributionArray | None = (
        np.empty((n_events, rates.shape[0])) if return_likelihood else None
    )

    for start in range(0, n_events, batch_size):
        stop = min(start + batch_size, n_events)
        batch_marks = marks[start:stop]
        predictive_batch = predictive_flat[time_ind[start:stop]]
        likelihood_batch = event_likelihood(rates[:, batch_marks].T)

        event_hpd[start:stop] = hpd_overlap(
            predictive_batch, likelihood_batch, coverage=coverage
        )
        event_kl[start:stop] = kl_divergence(predictive_batch, likelihood_batch)
        event_pvalue[start:stop] = mark_predictive_pvalue(predictive_batch, rates, batch_marks)
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


def baseline_threshold(baseline_values: ArrayLike, quantile: float) -> float:
    """Estimate a flagging threshold from baseline per-event diagnostic values.

    Returns the ``quantile`` of values pooled from a period (or simulation)
    where the model is believed to be well specified. Use a low quantile for
    diagnostics where small values indicate misfit (HPD overlap, e.g. 0.01)
    and a high quantile where large values do (KL divergence, e.g. 0.99), then
    flag events at or beyond the threshold (:func:`flag_events`). This is the
    paper's rule. NaN values are ignored.

    KL divergence is ``+inf`` when the prediction and the likelihood have
    disjoint support. Such values are allowed: when the requested quantile
    falls among them the threshold is ``+inf``, and then only infinite values
    are at or above it.

    Parameters
    ----------
    baseline_values : np.ndarray, shape (n_values,) or any shape
        Baseline diagnostic values; flattened before the quantile is taken.
    quantile : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    threshold : float
        The requested quantile (linear interpolation) of the baseline values.

    Raises
    ------
    ValueError
        If ``quantile`` is outside ``[0, 1]``, the baseline contains ``-inf``,
        or it has no finite values.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck import baseline_threshold
    >>> baseline_threshold(np.arange(101.0), 0.99)
    99.0
    """
    if not 0.0 <= quantile <= 1.0:
        msg = f"quantile must lie in [0, 1]; got {quantile}"
        raise ValueError(msg)
    values = np.asarray(baseline_values, dtype=float).ravel()
    if np.any(np.isneginf(values)):
        msg = "baseline_values contains -inf; a threshold cannot be estimated"
        raise ValueError(msg)
    values = values[~np.isnan(values)]
    if not np.any(np.isfinite(values)):
        msg = "baseline_values contains no finite values; the threshold would be undefined"
        raise ValueError(msg)
    # np.quantile interpolates linearly between the order statistics around the
    # quantile's position; interpolating toward +inf gives nan even with zero
    # weight, so check the two order statistics first.
    lower = np.quantile(values, quantile, method="lower")
    higher = np.quantile(values, quantile, method="higher")
    if lower == higher:
        return float(lower)
    if np.isinf(higher):
        return float(np.inf)
    return float(np.quantile(values, quantile))


class EventFlags(NamedTuple):
    """Per-event flags returned by :func:`flag_events`.

    Each field is a boolean array of shape ``(n_events,)`` marking events
    whose diagnostic is on the misfit side of its threshold, or ``None`` if
    no threshold was given for that diagnostic.

    Attributes
    ----------
    hpd_overlap : np.ndarray of bool, shape (n_events,), or None
        HPD overlap at or below its threshold.
    kl_divergence : np.ndarray of bool, shape (n_events,), or None
        KL divergence at or above its threshold.
    predictive_pvalue : np.ndarray of bool, shape (n_events,), or None
        Predictive p-value at or below its cutoff.
    """

    hpd_overlap: NDArray[np.bool_] | None
    kl_divergence: NDArray[np.bool_] | None
    predictive_pvalue: NDArray[np.bool_] | None


def flag_events(
    diagnostics: EventDiagnostics,
    *,
    hpd_overlap_threshold: float | None = None,
    kl_divergence_threshold: float | None = None,
    pvalue_threshold: float | None = 0.05,
) -> EventFlags:
    """Flag events whose diagnostics indicate poor local fit.

    Applies the paper's rule to each event independently: an event is flagged
    when its HPD overlap is **at or below** ``hpd_overlap_threshold``, its KL
    divergence is **at or above** ``kl_divergence_threshold``, or its
    predictive p-value is **at or below** ``pvalue_threshold``. Each
    diagnostic is flagged separately; NaN values are never flagged.

    Thresholds for HPD overlap and KL divergence depend on the model and the
    data, so they have no default. In its simulation the paper sets them from a
    period where the model is believed to fit, with :func:`baseline_threshold`
    (1st percentile of HPD overlap, 99th percentile of KL divergence); for real
    data without such a period it flags HPD overlap at a fixed 0.05 and sets no
    KL divergence cutoff. It flags p-values at a fixed 0.05. It recommends HPD overlap and the predictive p-value as the
    primary diagnostics and KL divergence as a reference, because KL
    divergence is also large for consistent events when the prediction is
    broad.

    Parameters
    ----------
    diagnostics : EventDiagnostics
        Per-event diagnostics from :func:`event_diagnostics`.
    hpd_overlap_threshold : float, optional
        Flag HPD overlap at or below this value. Default None (not flagged).
    kl_divergence_threshold : float, optional
        Flag KL divergence at or above this value. Default None (not flagged).
    pvalue_threshold : float, optional
        Flag predictive p-values at or below this value. Default 0.05; None
        skips the p-value.

    Returns
    -------
    EventFlags
        Boolean flags for each diagnostic that has a threshold, else None.

    Raises
    ------
    ValueError
        If a threshold is NaN.

    Examples
    --------
    Thresholds from a baseline period, as in the paper's simulation:

    >>> import numpy as np
    >>> from statespacecheck import baseline_threshold, event_diagnostics, flag_events
    >>> rng = np.random.default_rng(0)
    >>> predictive = rng.dirichlet(np.ones(20), size=200)  # (n_time, n_bins)
    >>> place_fields = rng.gamma(2.0, size=(20, 8))  # (n_bins, n_units)
    >>> time_ind, units = rng.integers(0, 200, 500), rng.integers(0, 8, 500)
    >>> diagnostics = event_diagnostics(predictive, place_fields, time_ind, units)
    >>> baseline = time_ind < 100
    >>> flags = flag_events(
    ...     diagnostics,
    ...     hpd_overlap_threshold=baseline_threshold(diagnostics.hpd_overlap[baseline], 0.01),
    ...     kl_divergence_threshold=baseline_threshold(
    ...         diagnostics.kl_divergence[baseline], 0.99
    ...     ),
    ... )
    >>> flags.predictive_pvalue.shape
    (500,)

    See Also
    --------
    baseline_threshold : Threshold from a baseline period
    event_diagnostics : Compute the per-event diagnostics
    """
    for name, threshold in (
        ("hpd_overlap_threshold", hpd_overlap_threshold),
        ("kl_divergence_threshold", kl_divergence_threshold),
        ("pvalue_threshold", pvalue_threshold),
    ):
        if threshold is not None:
            check_threshold_not_nan(threshold, name)
    hpd = np.asarray(diagnostics.hpd_overlap, dtype=float)
    kl = np.asarray(diagnostics.kl_divergence, dtype=float)
    pvalue = np.asarray(diagnostics.predictive_pvalue, dtype=float)
    # NaN compares False, so it is never flagged.
    return EventFlags(
        hpd_overlap=None if hpd_overlap_threshold is None else hpd <= hpd_overlap_threshold,
        kl_divergence=None
        if kl_divergence_threshold is None
        else kl >= kl_divergence_threshold,
        predictive_pvalue=None if pvalue_threshold is None else pvalue <= pvalue_threshold,
    )
