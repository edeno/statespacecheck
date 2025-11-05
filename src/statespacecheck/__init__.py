"""State space model goodness of fit diagnostics for neuroscience.

This package provides tools to assess the consistency between state
distributions and their component likelihood distributions in Bayesian
state space models.
"""

from statespacecheck.highest_density import highest_density_region
from statespacecheck.state_consistency import (
    hpd_overlap,
    kl_divergence,
)

__version__ = "0.1.0"

__all__ = [
    "highest_density_region",
    "kl_divergence",
    "hpd_overlap",
]
