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
- **[Interpreting the diagnostics](interpretation.md)**: what each diagnostic means,
  choosing thresholds, and turning flags into a modeling decision.
- **[Using your decoder](decoders.md)**: getting the inputs from a grid filter,
  `non_local_detector`, or a Kalman filter.
- **[API reference](reference/index.md)**: every function, grouped by task.
- **[Background tutorials](tutorials/index.md)**: highest-density regions, and the
  package's tools for time bins.

## Citation

--8<-- "README.md:citation"
