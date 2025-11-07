"""Analyze the simulation output to quantify misfit effects.

This script runs the simulation from sim.py and provides detailed quantitative
analysis of the diagnostic metrics across all phases.
"""

from __future__ import annotations

import numpy as np

# Import from the sim module in the same directory
try:
    from .sim import (
        DecodeParams,
        decode_and_diagnostics,
        gaussian_transition_matrix,
        simulate_spikes_flat_rate,
        simulate_spikes_position_tuned,
        simulate_walk,
    )
except ImportError:
    # Fallback for direct execution
    from sim import (
        DecodeParams,
        decode_and_diagnostics,
        gaussian_transition_matrix,
        simulate_spikes_flat_rate,
        simulate_spikes_position_tuned,
        simulate_walk,
    )


def analyze_simulation() -> None:
    """Run simulation and provide detailed analysis."""
    # Use current optimized parameters
    params = DecodeParams()
    assert params.pf_centers is not None

    print("=" * 80)
    print("SIMULATION ANALYSIS WITH RECOVERY PERIODS")
    print("=" * 80)
    print("\nSimulation timeline:")
    print(f"  0 - {params.T_remap_start:,}: Clean baseline")
    print(f"  {params.T_remap_start:,} - {params.T_remap_end:,}: Remapping misfit")
    print(f"  {params.T_remap_end:,} - {params.T_recovery1_end:,}: Recovery 1")
    print(f"  {params.T_recovery1_end:,} - {params.T_flat_end:,}: Flat firing misfit")
    print(f"  {params.T_flat_end:,} - {params.T_recovery2_end:,}: Recovery 2")
    print(f"  {params.T_recovery2_end:,} - {params.T_fast_end:,}: Fast movement misfit")
    print("\nParameters:")
    print(f"  sigx_pred (decoder): {params.sigx_pred}")
    print(f"  sigx_true_fast (fast phase): {params.sigx_true_fast}")
    print(f"  rate_scale: {params.rate_scale}")
    print("  flat_rate: 7e-3")
    n_remapped = len(params.remap_from_to) if isinstance(params.remap_from_to[0], tuple) else 1
    print(f"  Cells remapped: {n_remapped}")

    rng = np.random.default_rng(params.base_seed)

    # Grid & transition
    xs = np.arange(params.xs_min, params.xs_max + params.xs_step, params.xs_step, dtype=float)
    osm = gaussian_transition_matrix(xs, params.sigx_pred)

    # Generate all phases with recovery periods
    phases = []

    # Phase 1: Clean baseline (0 - T_remap_start)
    x_last = 0.0
    n_time = params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    x_last = x_true_phase[-1]

    # Phase 2: Remapping misfit (T_remap_start - T_remap_end)
    n_time = params.T_remap_end - params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    x_last = x_true_phase[-1]

    # Phase 3: Recovery 1 (T_remap_end - T_recovery1_end)
    n_time = params.T_recovery1_end - params.T_remap_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    x_last = x_true_phase[-1]

    # Phase 4: Flat firing misfit (T_recovery1_end - T_flat_end)
    n_time = params.T_flat_end - params.T_recovery1_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_flat_rate(n_time, len(params.pf_centers), rate=7e-3, rng=rng)
    phases.append((x_true_phase, spikes_phase))
    x_last = x_true_phase[-1]

    # Phase 5: Recovery 2 (T_flat_end - T_recovery2_end)
    n_time = params.T_recovery2_end - params.T_flat_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    x_last = x_true_phase[-1]

    # Phase 6: Fast movement misfit (T_recovery2_end - T_fast_end)
    n_time = params.T_fast_end - params.T_recovery2_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_true_fast, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))

    # Concatenate all phases
    spikes = np.vstack([s for _, s in phases])

    print("\nSpike statistics:")
    print(f"  Baseline mean spikes/cell/time: {np.mean(phases[0][1]):.4f}")
    print(f"  Remap mean spikes/cell/time: {np.mean(phases[1][1]):.4f}")
    print(f"  Recovery1 mean spikes/cell/time: {np.mean(phases[2][1]):.4f}")
    print(f"  Flat mean spikes/cell/time: {np.mean(phases[3][1]):.4f}")
    print(f"  Recovery2 mean spikes/cell/time: {np.mean(phases[4][1]):.4f}")
    print(f"  Fast mean spikes/cell/time: {np.mean(phases[5][1]):.4f}")

    # Decode
    print("\nRunning decoder...")
    metrics = decode_and_diagnostics(
        spikes=spikes,
        xs=xs,
        osm=osm,
        pf_centers=params.pf_centers,
        pf_width=params.pf_width,
        rate_scale=params.rate_scale,
        remap_window=params.remap_window,
        remap_from_to=params.remap_from_to,
    )

    # Define phase windows
    baseline_window = slice(0, params.T_remap_start)
    remap_window = slice(params.T_remap_start, params.T_remap_end)
    recovery1_window = slice(params.T_remap_end, params.T_recovery1_end)
    flat_window = slice(params.T_recovery1_end, params.T_flat_end)
    recovery2_window = slice(params.T_flat_end, params.T_recovery2_end)
    fast_window = slice(params.T_recovery2_end, params.T_fast_end)

    # Baseline statistics
    baseline_hpdo = metrics["HPDO"][baseline_window].ravel()
    baseline_kl = metrics["KL"][baseline_window].ravel()
    baseline_hpdo_mean = np.nanmean(baseline_hpdo)
    baseline_hpdo_std = np.nanstd(baseline_hpdo)
    baseline_kl_mean = np.nanmean(baseline_kl)
    baseline_kl_std = np.nanstd(baseline_kl)

    print("\n" + "=" * 80)
    print("PHASE-BY-PHASE ANALYSIS")
    print("=" * 80)

    print("\n--- BASELINE (Clean) ---")
    print(f"  HPD overlap: {baseline_hpdo_mean:.4f} ± {baseline_hpdo_std:.4f}")
    print(f"  KL divergence: {baseline_kl_mean:.4f} ± {baseline_kl_std:.4f}")

    # Analyze each phase
    phases_to_analyze = [
        ("REMAPPING MISFIT", remap_window),
        ("RECOVERY 1 (Clean)", recovery1_window),
        ("FLAT FIRING MISFIT", flat_window),
        ("RECOVERY 2 (Clean)", recovery2_window),
        ("FAST MOVEMENT MISFIT", fast_window),
    ]

    for phase_name, phase_slice in phases_to_analyze:
        phase_hpdo = metrics["HPDO"][phase_slice].ravel()
        phase_kl = metrics["KL"][phase_slice].ravel()

        hpdo_mean = np.nanmean(phase_hpdo)
        kl_mean = np.nanmean(phase_kl)

        hpdo_effect = (hpdo_mean - baseline_hpdo_mean) / (baseline_hpdo_std + 1e-10)
        kl_effect = (kl_mean - baseline_kl_mean) / (baseline_kl_std + 1e-10)

        print(f"\n--- {phase_name} ---")
        print(f"  HPD overlap: {hpdo_mean:.4f}")
        print(f"  KL divergence: {kl_mean:.4f}")
        print(f"  Effect size (HPD): {hpdo_effect:.2f} SD")
        print(f"  Effect size (KL): {kl_effect:.2f} SD")

        if "Clean" in phase_name or "RECOVERY" in phase_name:
            print(f"  → Recovery quality: {abs(hpdo_effect):.1f}σ from baseline")
        else:
            print(f"  → Misfit strength: {abs(hpdo_effect):.1f}σ drop in overlap")

    # Threshold violations
    hpdo_threshold = np.nanquantile(baseline_hpdo, 0.01)
    kl_threshold = np.nanquantile(baseline_kl, 0.99)

    print("\n" + "=" * 80)
    print("THRESHOLD VIOLATIONS")
    print("=" * 80)
    print("\nThresholds (from baseline):")
    print(f"  HPD overlap < {hpdo_threshold:.4f} (1st percentile)")
    print(f"  KL divergence > {kl_threshold:.4f} (99th percentile)")

    def violation_rate(data: np.ndarray, threshold: float, is_upper: bool) -> float:
        """Calculate percentage of non-NaN values violating threshold."""
        valid = ~np.isnan(data)
        if not valid.any():
            return 0.0
        if is_upper:
            violations = data[valid] > threshold
        else:
            violations = data[valid] < threshold
        return 100 * violations.sum() / valid.sum()

    for phase_name, phase_slice in phases_to_analyze:
        phase_hpdo = metrics["HPDO"][phase_slice].ravel()
        phase_kl = metrics["KL"][phase_slice].ravel()

        hpdo_viol = violation_rate(phase_hpdo, hpdo_threshold, is_upper=False)
        kl_viol = violation_rate(phase_kl, kl_threshold, is_upper=True)

        print(f"\n  {phase_name}:")
        print(f"    HPD overlap violations: {hpdo_viol:.1f}%")
        print(f"    KL divergence violations: {kl_viol:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_simulation()
