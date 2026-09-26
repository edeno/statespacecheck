"""State space model goodness of fit diagnostics for neuroscience.

This package provides tools to assess the consistency between state
distributions and their component likelihood distributions in Bayesian
state space models.
"""

from statespacecheck._validation import DistributionArray
from statespacecheck.events import (
    EventDiagnostics,
    EventFlags,
    baseline_threshold,
    event_diagnostics,
    event_likelihood,
    flag_events,
    mark_predictive_pvalue,
    predictive_mark_probabilities,
)
from statespacecheck.highest_density import DEFAULT_COVERAGE, highest_density_region
from statespacecheck.periods import (
    aggregate_over_period,
    combine_flags,
    find_low_overlap_intervals,
    flag_extreme_kl,
    flag_extreme_pvalues,
    flag_low_overlap,
)
from statespacecheck.predictive_checks import (
    log_predictive_density,
    predictive_density,
    predictive_pvalue,
)
from statespacecheck.state_consistency import (
    hpd_overlap,
    kl_divergence,
)
from statespacecheck.viz import plot_diagnostics

__all__ = [
    "DEFAULT_COVERAGE",
    "DistributionArray",
    "EventDiagnostics",
    "EventFlags",
    "__version__",
    "aggregate_over_period",
    "baseline_threshold",
    "combine_flags",
    "event_diagnostics",
    "event_likelihood",
    "find_low_overlap_intervals",
    "flag_events",
    "flag_extreme_kl",
    "flag_extreme_pvalues",
    "flag_low_overlap",
    "highest_density_region",
    "hpd_overlap",
    "kl_divergence",
    "log_predictive_density",
    "mark_predictive_pvalue",
    "plot_diagnostics",
    "predictive_density",
    "predictive_mark_probabilities",
    "predictive_pvalue",
]

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs
    from importlib.metadata import version

    __version__ = version("statespacecheck")
