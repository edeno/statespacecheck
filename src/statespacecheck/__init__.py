"""State space model goodness of fit diagnostics for neuroscience.

This package provides tools to assess the consistency between state
distributions and their component likelihood distributions in Bayesian
state space models.
"""

from statespacecheck._validation import DistributionArray
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
    "highest_density_region",
    "kl_divergence",
    "hpd_overlap",
    "predictive_density",
    "log_predictive_density",
    "aggregate_over_period",
    "predictive_pvalue",
    "DEFAULT_COVERAGE",
    "DistributionArray",
    "find_low_overlap_intervals",
    "flag_extreme_kl",
    "flag_extreme_pvalues",
    "flag_low_overlap",
    "combine_flags",
    "plot_diagnostics",
]

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs
    from importlib.metadata import version

    __version__ = version("statespacecheck")
