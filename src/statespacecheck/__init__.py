"""State space model goodness of fit diagnostics for neuroscience.

This package provides tools to assess the consistency between state
distributions and their component likelihood distributions in Bayesian
state space models.
"""

from statespacecheck.highest_density import highest_density_region
from statespacecheck.predictive_checks import (
    aggregate_over_period,
    log_predictive_density,
    predictive_density,
    predictive_pvalue,
)
from statespacecheck.state_consistency import (
    hpd_overlap,
    kl_divergence,
)

__version__ = "0.1.0"

# Default coverage probability for highest density regions
DEFAULT_COVERAGE = 0.95

__all__ = [
    "highest_density_region",
    "kl_divergence",
    "hpd_overlap",
    "predictive_density",
    "log_predictive_density",
    "aggregate_over_period",
    "predictive_pvalue",
    "DEFAULT_COVERAGE",
]
