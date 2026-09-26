"""Tests for visualization functions."""

from __future__ import annotations

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from statespacecheck.viz import plot_diagnostics


class TestPlotDiagnostics:
    """Test suite for plot_diagnostics function."""

    def test_basic_plot(self) -> None:
        """Test basic plot creation with valid inputs."""
        rng = np.random.default_rng(42)
        time = np.arange(100)
        overlap = rng.uniform(0.3, 0.9, 100)
        kl = rng.uniform(0.1, 2.0, 100)
        pvals = rng.uniform(0.1, 0.9, 100)

        fig = plot_diagnostics(time, overlap, kl, pvals)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 3 main axes + 1 twin axis for z(KL)
        plt.close(fig)

    def test_with_flags(self) -> None:
        """Test plot with flagged regions."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        pvals = np.ones(50) * 0.5
        flags = np.zeros(50, dtype=bool)
        flags[10:20] = True

        fig = plot_diagnostics(time, overlap, kl, pvals, flags=flags)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_flags(self) -> None:
        """Test plot without flags parameter."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        pvals = np.ones(50) * 0.5

        fig = plot_diagnostics(time, overlap, kl, pvals, flags=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_threshold_lines(self) -> None:
        """Each panel marks its flag threshold; the p-value panel has one line, at the cutoff."""
        time = np.arange(50)
        ones = np.ones(50)

        fig = plot_diagnostics(
            time,
            ones * 0.5,
            ones,
            ones * 0.5,
            overlap_threshold=0.3,
            kl_z_threshold=2.5,
            pvalue_threshold=0.01,
        )

        def hlines(ax: plt.Axes) -> list[float]:
            return [
                line.get_ydata()[0]
                for line in ax.get_lines()
                if len(set(line.get_ydata())) == 1 and len(line.get_ydata()) == 2
            ]

        overlap_ax, _, pvalue_ax, z_ax = fig.axes
        assert hlines(overlap_ax) == [0.3]
        assert hlines(z_ax) == [2.5]
        assert hlines(pvalue_ax) == [0.01]
        plt.close(fig)

    def test_metric_length_mismatch_error(self) -> None:
        """Metrics and time must have the same length."""
        with pytest.raises(ValueError, match="same length as time"):
            plot_diagnostics(np.arange(10), np.ones(10), np.ones(9), np.ones(10))

    def test_single_flagged_point_is_visible(self) -> None:
        """A run of one flagged sample is shaded with nonzero width."""
        time = np.arange(20.0)
        flags = np.zeros(20, dtype=bool)
        flags[7] = True

        fig = plot_diagnostics(time, np.ones(20), np.ones(20), np.ones(20), flags=flags)

        spans = list(fig.axes[0].patches)
        assert len(spans) == 1
        assert spans[0].get_extents().width > 0  # Polygon on matplotlib 3.8, Rectangle later
        plt.close(fig)

    def test_import_does_not_load_pyplot(self) -> None:
        """Importing the package does not import matplotlib.pyplot."""
        code = "import sys, statespacecheck; print('matplotlib.pyplot' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False"

    def test_with_nan_values(self) -> None:
        """Test plot handles NaN values."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        overlap[10] = np.nan
        kl = np.ones(50) * 1.0
        kl[15] = np.nan
        pvals = np.ones(50) * 0.5
        pvals[20] = np.nan

        fig = plot_diagnostics(time, overlap, kl, pvals)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_inf_values(self) -> None:
        """Test plot handles Inf values."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        kl[15] = np.inf
        pvals = np.ones(50) * 0.5

        fig = plot_diagnostics(time, overlap, kl, pvals)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_flags_shape_mismatch_error(self) -> None:
        """Test that mismatched flags shape raises ValueError."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        pvals = np.ones(50) * 0.5
        flags = np.zeros(30, dtype=bool)  # Wrong length

        with pytest.raises(ValueError, match="flags must have the same length as time"):
            plot_diagnostics(time, overlap, kl, pvals, flags=flags)

    def test_three_panel_structure(self) -> None:
        """Test that plot has correct three-panel structure."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        pvals = np.ones(50) * 0.5

        fig = plot_diagnostics(time, overlap, kl, pvals)

        # Should have 3 main subplots + 1 twin y-axis for KL
        axes = fig.get_axes()
        # Should have 4 axes total: 3 main + 1 twin
        assert len(axes) == 4
        plt.close(fig)

    def test_axis_labels(self) -> None:
        """Test that axes have correct labels."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        pvals = np.ones(50) * 0.5

        fig = plot_diagnostics(time, overlap, kl, pvals)
        axes = fig.get_axes()

        # Check for expected label content (case-insensitive)
        labels = [ax.get_ylabel().lower() for ax in axes]
        # Should have overlap, KL, and p-value related labels
        assert any("overlap" in label for label in labels)
        assert any("kl" in label or "divergence" in label for label in labels)
        assert any("p" in label or "predictive" in label for label in labels)

        plt.close(fig)

    def test_empty_flags(self) -> None:
        """Test with flags array that has no True values."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        pvals = np.ones(50) * 0.5
        flags = np.zeros(50, dtype=bool)  # All False

        fig = plot_diagnostics(time, overlap, kl, pvals, flags=flags)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_flags_true(self) -> None:
        """Test with flags array that has all True values."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        pvals = np.ones(50) * 0.5
        flags = np.ones(50, dtype=bool)  # All True

        fig = plot_diagnostics(time, overlap, kl, pvals, flags=flags)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nonuniform_time(self) -> None:
        """Test with non-uniform time spacing."""
        time = np.array([0, 1, 2, 5, 10, 20, 50])
        overlap = np.ones(7) * 0.5
        kl = np.ones(7) * 1.0
        pvals = np.ones(7) * 0.5

        fig = plot_diagnostics(time, overlap, kl, pvals)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_timepoint(self) -> None:
        """Test with single time point."""
        time = np.array([0])
        overlap = np.array([0.5])
        kl = np.array([1.0])
        pvals = np.array([0.5])

        fig = plot_diagnostics(time, overlap, kl, pvals)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_robust_zscore_displayed(self) -> None:
        """Test that robust z-score is displayed on secondary axis."""
        time = np.arange(50)
        overlap = np.ones(50) * 0.5
        kl = np.ones(50) * 1.0
        kl[20:25] = 10.0  # Create spike for z-score
        pvals = np.ones(50) * 0.5

        fig = plot_diagnostics(time, overlap, kl, pvals)

        # Check that we have a twin y-axis for z-scores
        axes = fig.get_axes()
        assert len(axes) > 3  # Should have more than 3 due to twin axis

        plt.close(fig)


def test_datetime_time_axis_with_flags() -> None:
    """Flagged runs are shaded on a datetime time axis too."""
    time = np.datetime64("2026-01-01T00:00:00") + np.arange(20) * np.timedelta64(1, "s")
    flags = np.zeros(20, dtype=bool)
    flags[5:8] = True
    fig = plot_diagnostics(time, np.ones(20), np.ones(20), np.ones(20), flags=flags)
    assert len(fig.axes[0].patches) == 1
    plt.close(fig)
