"""Complete workshop solution for the ACF Gaussian-sum validity analysis.

This script solves the full assignment from ``Taller_acf.pdf``:
1. Theoretical validity condition from Wiener-Khinchin.
2. Numerical verification with FFT-based spectral estimation.
3. Toeplitz positive-semidefinite checks for sampled lags.
4. Plots of R(tau) and S(f) for multiple parameter configurations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ACFConfiguration:
    """Stores one parameter configuration for the candidate ACF model."""

    name: str
    a0: float
    a1: float
    sigma_s: float


def validate_configuration(
    cfg: ACFConfiguration,  # Candidate ACF parameter configuration
) -> None:  # Raises ValueError when parameters are outside valid ranges
    """Validates positivity assumptions required by the workshop model.

    Purpose:
        Enforce the statement constraints ``A_k > 0`` and ``sigma > 0`` for
        each tested configuration before any numerical analysis.

    Parameters:
        cfg: One model configuration containing A0, A1 and sigma.

    Returns:
        No return value. Raises ``ValueError`` on invalid parameters.

    Side effects:
        None.

    Assumptions:
        The model is restricted to k in {-1, 0, 1} with A_{-1}=A_1=a1.
    """
    if cfg.a0 <= 0.0:
        raise ValueError(f"{cfg.name}: a0 must be strictly positive.")
    if cfg.a1 <= 0.0:
        raise ValueError(f"{cfg.name}: a1 must be strictly positive.")
    if cfg.sigma_s <= 0.0:
        raise ValueError(f"{cfg.name}: sigma_s must be strictly positive.")


def acf_candidate(
    tau_s: NDArray[np.float64],  # Lag axis where R(tau) is sampled [s]
    cfg: ACFConfiguration,  # ACF parameter configuration
) -> NDArray[np.float64]:  # Candidate autocorrelation samples R(tau)
    """Evaluates R(tau) for k in {-1, 0, 1} with A_{-1}=A_1.

    Purpose:
        Build the workshop candidate:
        R(tau) = sum_{k=-1,0,1} A_k exp(-(tau-k)^2/(2*sigma_k^2)),
        with sigma_k^2 = sigma^2 for k in {-1, 0, 1}.

    Parameters:
        tau_s: Lag axis in seconds used for numerical sampling.
        cfg: Model parameters (a0, a1, sigma_s).

    Returns:
        Array with R(tau) evaluated on ``tau_s``.

    Side effects:
        None.
    """
    sigma2_s2: float = cfg.sigma_s**2

    # Build the central and shifted Gaussian terms explicitly to preserve
    # one-to-one correspondence with the analytical model.
    central_term = cfg.a0 * np.exp(-(tau_s**2) / (2.0 * sigma2_s2))
    right_shift_term = cfg.a1 * np.exp(-((tau_s - 1.0) ** 2) / (2.0 * sigma2_s2))
    left_shift_term = cfg.a1 * np.exp(-((tau_s + 1.0) ** 2) / (2.0 * sigma2_s2))
    return central_term + right_shift_term + left_shift_term


def acf_spectrum_closed_form(
    frequency_hz: NDArray[np.float64],  # Frequency axis [Hz]
    cfg: ACFConfiguration,  # ACF parameter configuration
) -> NDArray[np.float64]:  # Closed-form spectrum S(f)
    """Evaluates the analytical Fourier transform S(f) of the candidate ACF.

    Purpose:
        Apply the closed-form transform of shifted Gaussians to derive:
        S(f) = sqrt(2*pi)*sigma*exp(-2*pi^2*sigma^2*f^2) * (a0 + 2*a1*cos(2*pi*f)).
        This expression is the key Wiener-Khinchin validity check.

    Parameters:
        frequency_hz: Frequency axis in hertz.
        cfg: Model parameters (a0, a1, sigma_s).

    Returns:
        Analytical real-valued spectrum samples S(f).

    Side effects:
        None.
    """
    gaussian_envelope = (
        np.sqrt(2.0 * np.pi)
        * cfg.sigma_s
        * np.exp(-2.0 * np.pi**2 * (cfg.sigma_s**2) * (frequency_hz**2))
    )
    cosine_factor = cfg.a0 + 2.0 * cfg.a1 * np.cos(2.0 * np.pi * frequency_hz)
    return gaussian_envelope * cosine_factor


def acf_spectrum_fft(
    tau_s: NDArray[np.float64],  # Lag axis where R(tau) is sampled [s]
    r_tau: NDArray[np.float64],  # ACF samples aligned with tau_s
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # (f_hz, Re{FFT[R]})
    """Estimates S(f) numerically from sampled R(tau) using FFT.

    Purpose:
        Approximate the continuous Fourier transform with a Riemann sum:
        S(f) ≈ delta_tau * FFT{R(tau)} after consistent centering/ordering.

    Parameters:
        tau_s: Uniform lag grid in seconds.
        r_tau: ACF samples on ``tau_s``.

    Returns:
        frequency_hz: Centered frequency axis in hertz.
        spectrum_real: Real part of the FFT-based spectrum estimate.

    Side effects:
        None.

    Assumptions:
        ``tau_s`` is uniformly spaced and ``r_tau`` is sampled on the same grid.
    """
    if tau_s.ndim != 1 or r_tau.ndim != 1:
        raise ValueError("tau_s and r_tau must be one-dimensional arrays.")
    if tau_s.size != r_tau.size:
        raise ValueError("tau_s and r_tau must have the same length.")
    if tau_s.size < 3:
        raise ValueError("At least 3 lag samples are required.")

    delta_tau_s: float = float(tau_s[1] - tau_s[0])
    if delta_tau_s <= 0.0:
        raise ValueError("tau_s must be strictly increasing.")

    # Compute FFT with shift conventions so that tau=0 maps to centered spectrum.
    spectrum_complex = delta_tau_s * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(r_tau)))
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(tau_s.size, d=delta_tau_s))
    return frequency_hz.astype(np.float64), np.real(spectrum_complex).astype(np.float64)


def build_toeplitz_from_acf(
    r_nonnegative: NDArray[np.float64],  # ACF samples for lags 0..K
    matrix_size: int,  # Desired Toeplitz matrix size N
) -> NDArray[np.float64]:  # Toeplitz matrix T_ij = R(|i-j|)
    """Builds a Toeplitz matrix from nonnegative-lag ACF samples.

    Purpose:
        Numerically verify positive semidefiniteness in finite dimensions,
        equivalent to covariance-matrix validity checks for WSS processes.

    Parameters:
        r_nonnegative: ACF values at lags [0, 1, ..., K] on a uniform grid.
        matrix_size: Size N of the Toeplitz covariance matrix.

    Returns:
        NxN Toeplitz matrix built from absolute lag differences.

    Side effects:
        None.
    """
    if matrix_size < 2:
        raise ValueError("matrix_size must be at least 2.")
    if r_nonnegative.ndim != 1:
        raise ValueError("r_nonnegative must be one-dimensional.")
    if r_nonnegative.size < matrix_size:
        raise ValueError("r_nonnegative length must be >= matrix_size.")

    # Fill T[i, j] = R(|i-j|) using vectorized index differences.
    row_idx, col_idx = np.indices((matrix_size, matrix_size))
    lag_index = np.abs(row_idx - col_idx)
    return r_nonnegative[lag_index]


def theoretical_validity_condition(
    cfg: ACFConfiguration,  # ACF parameter configuration
) -> bool:  # True when the Wiener-Khinchin nonnegativity condition holds
    """Evaluates the exact analytical condition for PSD nonnegativity.

    Purpose:
        Decide validity using the derived inequality:
        S(f) >= 0 for all f  <=>  a0 - 2*a1 >= 0.

    Parameters:
        cfg: Model parameters (a0, a1, sigma_s).

    Returns:
        True when the candidate can be a valid real WSS ACF.

    Side effects:
        None.
    """
    return (cfg.a0 - 2.0 * cfg.a1) >= 0.0


def analyze_configuration(
    cfg: ACFConfiguration,  # ACF parameter configuration
    tau_s: NDArray[np.float64],  # Lag axis used for analysis [s]
) -> dict[str, float | int | bool | str]:  # Structured metrics for one setup
    """Computes structural and spectral validity metrics for one configuration.

    Purpose:
        Produce quantitative evidence for workshop conclusions:
        symmetry, maximum-at-zero, spectral nonnegativity, and Toeplitz PSDness.

    Parameters:
        cfg: Model parameters under test.
        tau_s: Lag axis in seconds for numerical sampling.

    Returns:
        Dictionary with scalar metrics and boolean decisions.

    Side effects:
        None.
    """
    validate_configuration(cfg)

    # Evaluate ACF on symmetric lags and verify Hermitian symmetry numerically.
    r_tau = acf_candidate(tau_s=tau_s, cfg=cfg)
    r_minus_tau = acf_candidate(tau_s=-tau_s, cfg=cfg)
    symmetry_error = float(np.max(np.abs(r_tau - r_minus_tau)))

    # Compare R(0) against global sampled maximum for the max-at-zero property.
    idx_zero = int(np.argmin(np.abs(tau_s)))
    r0_value = float(r_tau[idx_zero])
    idx_global_max = int(np.argmax(r_tau))
    r_global_max = float(r_tau[idx_global_max])
    tau_at_max_s = float(tau_s[idx_global_max])
    max_violation = float(r_global_max - r0_value)

    # Compute analytical and FFT spectra, then count negative samples.
    frequency_hz_closed = np.linspace(-2.5, 2.5, 8001, dtype=np.float64)
    spectrum_closed = acf_spectrum_closed_form(frequency_hz=frequency_hz_closed, cfg=cfg)
    frequency_hz_fft, spectrum_fft = acf_spectrum_fft(tau_s=tau_s, r_tau=r_tau)

    min_spectrum_closed = float(np.min(spectrum_closed))
    min_spectrum_fft = float(np.min(spectrum_fft))
    negative_closed_count = int(np.count_nonzero(spectrum_closed < -1e-10))
    negative_fft_count = int(np.count_nonzero(spectrum_fft < -1e-6))

    # Build finite Toeplitz covariance matrix and inspect smallest eigenvalue.
    idx_nonnegative = tau_s >= -1e-12
    r_nonnegative = r_tau[idx_nonnegative]
    toeplitz_matrix = build_toeplitz_from_acf(r_nonnegative=r_nonnegative, matrix_size=80)
    min_toeplitz_eig = float(np.min(np.linalg.eigvalsh(toeplitz_matrix)))

    return {
        "name": cfg.name,
        "a0": cfg.a0,
        "a1": cfg.a1,
        "sigma_s": cfg.sigma_s,
        "a0_minus_2a1": cfg.a0 - 2.0 * cfg.a1,
        "symmetry_error": symmetry_error,
        "r0_value": r0_value,
        "r_global_max": r_global_max,
        "tau_at_global_max_s": tau_at_max_s,
        "max_violation_rmax_minus_r0": max_violation,
        "min_spectrum_closed": min_spectrum_closed,
        "min_spectrum_fft": min_spectrum_fft,
        "negative_closed_count": negative_closed_count,
        "negative_fft_count": negative_fft_count,
        "min_toeplitz_eigenvalue": min_toeplitz_eig,
        "theoretical_validity": theoretical_validity_condition(cfg=cfg),
    }


def plot_configuration(
    cfg: ACFConfiguration,  # ACF parameter configuration
    tau_s: NDArray[np.float64],  # Lag axis [s]
    output_dir: Path,  # Directory where plots are saved
) -> Path:  # Path to the saved figure file
    """Creates and saves R(tau) and S(f) plots for one configuration.

    Purpose:
        Generate the workshop-required graphics for the time and frequency
        domains, explicitly marking negative spectral regions when present.

    Parameters:
        cfg: Model parameters under test.
        tau_s: Lag axis in seconds.
        output_dir: Destination directory for image files.

    Returns:
        Full path to the saved PNG figure.

    Side effects:
        Writes one PNG file to ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate ACF and both spectral versions for side-by-side inspection.
    r_tau = acf_candidate(tau_s=tau_s, cfg=cfg)
    frequency_hz_fft, spectrum_fft = acf_spectrum_fft(tau_s=tau_s, r_tau=r_tau)
    frequency_hz_closed = np.linspace(-2.5, 2.5, 8001, dtype=np.float64)
    spectrum_closed = acf_spectrum_closed_form(frequency_hz=frequency_hz_closed, cfg=cfg)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    # Left panel: ACF with explicit mirrored curve to show even symmetry.
    axes[0].plot(tau_s, r_tau, lw=2.2, label=r"$R(\tau)$")
    axes[0].plot(tau_s, acf_candidate(tau_s=-tau_s, cfg=cfg), "--", lw=1.6, label=r"$R(-\tau)$")
    axes[0].set_title(
        f"ACF Candidate - {cfg.name}\n"
        f"$A_0={cfg.a0:.2f}$, $A_1={cfg.a1:.2f}$, $\\sigma={cfg.sigma_s:.2f}$ s"
    )
    axes[0].set_xlabel(r"Lag $\tau$ [s]")
    axes[0].set_ylabel(r"$R(\tau)$")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    # Right panel: analytical PSD plus FFT estimate, with negative zones emphasized.
    negative_closed = np.where(spectrum_closed < 0.0, spectrum_closed, np.nan)
    axes[1].plot(frequency_hz_closed, spectrum_closed, lw=2.2, label="Analytical $S(f)$")
    axes[1].plot(frequency_hz_fft, spectrum_fft, lw=1.2, alpha=0.8, label="FFT estimate")
    axes[1].plot(frequency_hz_closed, negative_closed, lw=2.4, color="red", label="Negative region")
    axes[1].axhline(0.0, color="0.4", lw=1.0, linestyle=":")
    axes[1].set_title("PSD via Wiener-Khinchin")
    axes[1].set_xlabel("Frequency $f$ [Hz]")
    axes[1].set_ylabel("$S(f)$")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    figure_path = output_dir / f"{cfg.name.lower().replace(' ', '_')}.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_summary_markdown(
    metrics: list[dict[str, float | int | bool | str]],  # Per-case computed metrics
    figure_paths: list[Path],  # PNG paths generated by plot_configuration
    output_path: Path,  # Markdown report destination
) -> None:  # Writes a concise computational summary table
    """Writes a compact markdown summary of numerical results.

    Purpose:
        Persist reproducible numerical evidence (symmetry, PSD checks, Toeplitz
        eigenvalues) to support the theoretical workshop conclusion.

    Parameters:
        metrics: List of metrics dictionaries, one per configuration.
        figure_paths: Saved figure paths aligned with ``metrics`` order.
        output_path: Target markdown file path.

    Returns:
        No return value.

    Side effects:
        Writes one markdown file to disk.
    """
    lines: list[str] = []
    lines.append("# Taller ACF - Resumen Computacional\n")
    lines.append(
        "Criterio teórico principal: la ACF es válida para WSS real solo si "
        "`A0 - 2*A1 >= 0`.\n"
    )
    lines.append(
        "| Caso | A0 | A1 | sigma [s] | A0-2A1 | min S(f) cerrada | min S(f) FFT | "
        "negativos cerrada | min eig Toeplitz | ¿válida teóricamente? |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")

    for row in metrics:
        lines.append(
            "| "
            f"{row['name']} | {row['a0']:.3f} | {row['a1']:.3f} | {row['sigma_s']:.3f} | "
            f"{row['a0_minus_2a1']:.3f} | {row['min_spectrum_closed']:.6e} | "
            f"{row['min_spectrum_fft']:.6e} | {row['negative_closed_count']} | "
            f"{row['min_toeplitz_eigenvalue']:.6e} | "
            f"{'SI' if bool(row['theoretical_validity']) else 'NO'} |\n"
        )

    lines.append("\n## Figuras\n")
    for path in figure_paths:
        lines.append(f"- `{path.as_posix()}`\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    """Runs the complete numerical part of the workshop and writes artifacts."""
    output_dir = Path("output/taller_acf")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a wide lag window and dense sampling for stable FFT approximation.
    tau_s = np.linspace(-6.0, 6.0, 12001, dtype=np.float64)

    configurations = [
        ACFConfiguration(name="Caso 1 Valido", a0=2.8, a1=1.0, sigma_s=0.35),
        ACFConfiguration(name="Caso 2 Umbral", a0=2.0, a1=1.0, sigma_s=0.65),
        ACFConfiguration(name="Caso 3 Invalido", a0=1.2, a1=1.0, sigma_s=0.90),
    ]

    metrics: list[dict[str, float | int | bool | str]] = []
    figure_paths: list[Path] = []

    # Analyze each case and generate the required R(tau) and S(f) graphics.
    for cfg in configurations:
        row = analyze_configuration(cfg=cfg, tau_s=tau_s)
        metrics.append(row)
        figure_paths.append(plot_configuration(cfg=cfg, tau_s=tau_s, output_dir=output_dir))

    # Persist machine-readable metrics for reproducibility.
    metrics_json_path = output_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")

    # Write a short markdown summary that complements the full theoretical report.
    write_summary_markdown(
        metrics=metrics,
        figure_paths=figure_paths,
        output_path=output_dir / "summary.md",
    )

    print("Generated workshop artifacts:")
    print(f"- Metrics JSON: {metrics_json_path.as_posix()}")
    print(f"- Summary MD: {(output_dir / 'summary.md').as_posix()}")
    for figure_path in figure_paths:
        print(f"- Figure: {figure_path.as_posix()}")


if __name__ == "__main__":
    main()
