# Tutorials

These notebooks are rendered with their outputs, so they can be read without running
anything.

## Start here

### [5. Per-spike diagnostics: the paper's workflow](05_per_event_diagnostics.ipynb)

The workflow of *Local goodness-of-fit measures for neural decoding*, at small scale:
decode position from simulated place cells, compute HPD overlap, the predictive
p-value and KL divergence for every spike, set thresholds from a baseline period, find
when and for which units the model fails, and check whether a revised model fixes it.
Needs only `pip install statespacecheck matplotlib`.

## Background and extensions

These tutorials explain the building blocks and the package's tools beyond the paper.
They work with whole time bins rather than individual spikes.

### [1. Introduction](01_introduction.ipynb)

What a state space model's prediction and an observation's likelihood are, and how KL
divergence and HPD overlap compare them, on simple Gaussian examples.

### [2. Highest density regions](02_highest_density_regions.ipynb)

How highest-density regions are computed, including for multimodal and 2-D
distributions, and how coverage affects them.

### [3. Time-resolved diagnostics](03_time_resolved_diagnostics.ipynb)

Diagnostics over the time bins of a session, with the run-based flagging functions of
`statespacecheck.periods` (an extension beyond the paper).

### [4. Predictive checks](04_predictive_checks.ipynb)

Monte Carlo predictive checks of whole time bins with `log_predictive_density` and
`predictive_pvalue` (an extension beyond the paper, whose p-value is per spike).

## Running the tutorials

Each tutorial is a pair of files in the repository's
[`examples/`](https://github.com/edeno/statespacecheck/tree/main/examples) directory:
a notebook (`.ipynb`) with its outputs and the same code as a script (`.py`).

- **Tutorial 5** is self-contained: download the notebook and run it anywhere with
  `statespacecheck` and `matplotlib` installed, for example in
  [Google Colab](https://colab.research.google.com/) after
  `!pip install statespacecheck`.
- **Tutorials 1–4** import helpers from `examples/utils.py`, so run them from a clone
  of the repository:

  ```bash
  git clone https://github.com/edeno/statespacecheck.git
  cd statespacecheck
  uv sync --extra docs
  uv run --extra docs jupyter nbconvert --to notebook --execute examples/01_introduction.ipynb
  ```

  or open the notebooks in any Jupyter front end (JupyterLab, VS Code) using that
  environment.

Found a problem? Please [open an issue](https://github.com/edeno/statespacecheck/issues).
