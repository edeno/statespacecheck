# statespacecheck

**Local goodness-of-fit diagnostics for state space models: find the observations, down
to individual spikes, where a decoder disagrees with the data.**

--8<-- "README.md:intro"

## Installation

```bash
pip install statespacecheck
```

## Quick start

--8<-- "README.md:quickstart"

--8<-- "README.md:reading"

## Where to go next

- **[Per-spike diagnostics: the paper's workflow](tutorials/05_per_event_diagnostics.ipynb)**:
  a complete example, from decoding to finding which units a model fails for.
- **[Clusterless per-spike diagnostics](tutorials/06_clusterless_diagnostics.ipynb)**:
  the same diagnostics for spikes described by waveform features rather than units.
- **[Interpreting the diagnostics](interpretation.md)**: what each diagnostic means,
  choosing thresholds, and turning flags into a modeling decision.
- **[Using your decoder](decoders.md)**: getting the inputs from a grid filter,
  `non_local_detector`, or a Kalman filter.
- **[API reference](reference/index.md)**: every function, grouped by task.
- **[A general state space model](tutorials/07_kalman_filter.ipynb)**: a Kalman
  filter on a non-neural tracking problem, and how to put continuous distributions
  on the grid of states the diagnostics take.
- **[Background tutorials](tutorials/index.md)**: highest-density regions, and the
  package's tools for time bins.

## Citation

--8<-- "README.md:citation"
