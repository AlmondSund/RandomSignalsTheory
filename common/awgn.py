"""AWGN utilities for robust, validated signal perturbation."""

import warnings
from typing import Optional

import numpy as np


def _validate_snr_db(
    snr_db: float,  # Requested signal-to-noise ratio [dB]
) -> float:  # Validated SNR in decibels
    """Validates and normalizes the requested SNR in decibels."""
    # Convert to a scalar to reject non-numeric or array-like inputs early.
    try:
        snr_db_value = float(snr_db)
    except (TypeError, ValueError) as exc:
        raise ValueError("snr_db must be a real scalar in decibels [dB].") from exc

    if np.isnan(snr_db_value):
        raise ValueError("snr_db must not be NaN.")

    return snr_db_value


def _is_quantizing_dtype(
    dtype: np.dtype,  # Candidate dtype for the output signal
) -> bool:  # True when casting back would quantize AWGN heavily
    """Returns whether preserving this dtype would quantize or erase AWGN."""
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)


def _resolve_output_dtype(
    x_dtype: np.dtype,  # Input dtype
    is_complex: bool,  # Whether the signal is complex-valued
    preserve_dtype: bool,  # Whether output must keep the input dtype
) -> np.dtype:  # Chosen output dtype for y = x + n
    """Resolves the output dtype according to preserve_dtype semantics."""
    if preserve_dtype:
        return np.dtype(x_dtype)
    return np.dtype(np.complex128 if is_complex else np.float64)


def _compute_signal_power(
    x_t: np.ndarray,  # Input signal samples (real or complex)
) -> float:  # Average discrete-time signal power E[|x|^2]
    """Computes signal power robustly using promoted precision.

    Parameters
    ----------
    x_t : np.ndarray
        Input signal samples.

    Returns
    -------
    float
        Average signal power E[|x|^2] as a finite non-negative scalar.

    Side Effects
    ------------
    None.
    """
    # Promote before multiplication to prevent overflow for integer inputs.
    if np.iscomplexobj(x_t):
        x_for_power = x_t.astype(np.complex128, copy=False)
        # Use x*conj(x) to avoid abs/exponent temporaries for complex inputs.
        power_samples = (x_for_power * np.conjugate(x_for_power)).real
    else:
        x_for_power = x_t.astype(np.float64, copy=False)
        # Use x*x to avoid abs/exponent temporaries for real inputs.
        power_samples = x_for_power * x_for_power

    signal_power = float(np.mean(power_samples, dtype=np.float64))
    if not np.isfinite(signal_power):
        raise ValueError("x_t power is non-finite after dtype promotion; check signal magnitude.")
    return signal_power


def _compute_noise_std(
    signal_power: float,  # Signal power E[|x|^2]
    snr_db: float,  # Requested signal-to-noise ratio [dB]
) -> float:  # AWGN standard deviation for real-valued components
    """Converts SNR [dB] to a validated AWGN standard deviation.

    Parameters
    ----------
    signal_power : float
        Average signal power E[|x|^2].
    snr_db : float
        Requested SNR in decibels [dB].

    Returns
    -------
    float
        Standard deviation used for real AWGN components.

    Side Effects
    ------------
    None.
    """
    # Derive linear SNR and noise power while keeping overflow/underflow silent.
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        snr_linear = np.power(10.0, snr_db / 10.0)
        noise_power = np.divide(signal_power, snr_linear)
        noise_std = np.sqrt(noise_power)

    # Validate derived parameters while allowing legitimate zero-noise limits.
    if not np.isfinite(noise_power) or noise_power < 0.0:
        raise ValueError("Derived noise_power is invalid; snr_db is too extreme for stable generation.")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("Derived noise_std is invalid; snr_db is too extreme for stable generation.")

    return float(noise_std)


def _sample_standard_normal(
    shape: tuple[int, ...],  # Desired shape of random samples
    dtype: np.dtype,  # Floating dtype for the samples
    rng: np.random.Generator,  # Random generator used for sampling
) -> np.ndarray:  # Standard normal samples N(0, 1)
    """Samples N(0, 1) with a preferred dtype while keeping compatibility.

    Parameters
    ----------
    shape : tuple[int, ...]
        Target shape.
    dtype : np.dtype
        Preferred floating dtype for generated samples.
    rng : np.random.Generator
        Generator from which random numbers are consumed.

    Returns
    -------
    np.ndarray
        Standard normal samples with the requested shape and practical dtype.

    Side Effects
    ------------
    Consumes random numbers from ``rng``.
    """
    preferred_dtype = np.dtype(dtype)
    if preferred_dtype == np.float64:
        return rng.standard_normal(size=shape)

    # Use dtype-aware generation when available, then fall back to casting.
    try:
        return rng.standard_normal(size=shape, dtype=preferred_dtype)
    except TypeError:
        return rng.standard_normal(size=shape).astype(preferred_dtype, copy=False)


def _draw_white_noise(
    shape: tuple[int, ...],  # Desired shape for the sampled noise
    noise_std: float,  # Standard deviation for real noise components
    is_complex: bool,  # Whether to generate complex-valued noise
    rng: np.random.Generator,  # Random generator used to sample noise
    output_dtype: np.dtype,  # Final dtype for returned noise samples
) -> np.ndarray:  # White Gaussian noise samples
    """Draws white Gaussian noise in the requested output dtype.

    Parameters
    ----------
    shape : tuple[int, ...]
        Target shape.
    noise_std : float
        Standard deviation for real-valued components.
    is_complex : bool
        True for complex AWGN generation.
    rng : np.random.Generator
        Generator from which random numbers are consumed.
    output_dtype : np.dtype
        Final dtype for the returned noise array.

    Returns
    -------
    np.ndarray
        White Gaussian noise with the requested shape.

    Side Effects
    ------------
    Consumes random numbers from ``rng``.
    """
    output_dtype_obj = np.dtype(output_dtype)

    # For complex AWGN each component gets half the total variance.
    if is_complex:
        component_dtype = np.float32 if output_dtype_obj == np.complex64 else np.float64
        component_std = np.asarray(noise_std / np.sqrt(2.0), dtype=component_dtype)
        noise_real = _sample_standard_normal(shape=shape, dtype=component_dtype, rng=rng) * component_std
        noise_imag = _sample_standard_normal(shape=shape, dtype=component_dtype, rng=rng) * component_std
        noise = noise_real + 1j * noise_imag
        return noise.astype(output_dtype_obj, copy=False)

    component_dtype = np.float32 if output_dtype_obj == np.float16 else output_dtype_obj
    scale = np.asarray(noise_std, dtype=component_dtype)
    noise_real = _sample_standard_normal(shape=shape, dtype=component_dtype, rng=rng) * scale
    return noise_real.astype(output_dtype_obj, copy=False)


def awgn(
    x_t: np.ndarray,  # Input signal samples (real or complex)
    snr_db: float,  # Target SNR with respect to mean signal power [dB]
    rng: Optional[np.random.Generator] = None,  # RNG for reproducible noise generation
    preserve_dtype: bool = True,  # If True, cast output back to x_t.dtype
) -> np.ndarray:  # Noisy signal with the same shape as x_t
    """Adds additive white Gaussian noise (AWGN) to a discrete-time signal.

    Purpose
    -------
    Perturbs a real or complex signal with zero-mean white Gaussian noise so
    that the target signal-to-noise ratio matches ``snr_db``.

    Parameters
    ----------
    x_t : np.ndarray
        Input signal samples. Power is estimated as E[|x|^2] on the array.
    snr_db : float
        Desired signal-to-noise ratio in decibels [dB].
        ``np.inf`` means no noise is added.
    rng : Optional[np.random.Generator], default=None
        Random number generator used for noise sampling.
    preserve_dtype : bool, default=True
        If True, cast output back to ``x_t.dtype``.
        If False, return ``float64`` for real inputs and ``complex128`` for
        complex inputs. For finite ``snr_db``, integer/bool inputs with
        ``preserve_dtype=True`` are rejected to prevent quantized AWGN.
        ``float16`` is allowed but may heavily quantize low-amplitude noise,
        so a runtime warning is emitted in that configuration.
        ``complex64`` is allowed; at extreme finite SNR values, limited
        dynamic range can increase overflow/saturation risk.

    Returns
    -------
    np.ndarray
        Noisy signal with the same shape as ``x_t``.

    Side Effects
    ------------
    Consumes random numbers from ``rng`` or from a newly created generator.

    Assumptions
    -----------
    - ``x_t`` contains at least one sample.
    - ``x_t`` has non-zero average power unless ``snr_db == np.inf``.
    - ``snr_db`` is real and not NaN.
    - ``snr_db = -np.inf`` is invalid because it implies infinite noise power.
    """
    x = np.asarray(x_t)
    snr_db_value = _validate_snr_db(snr_db)
    is_complex = np.iscomplexobj(x)
    output_dtype = _resolve_output_dtype(x.dtype, is_complex, preserve_dtype)

    # Guard against degenerate inputs before power and SNR calculations.
    if x.size == 0:
        raise ValueError("x_t must contain at least one sample.")
    if np.isneginf(snr_db_value):
        raise ValueError("snr_db = -inf is not supported because it implies infinite noise power.")
    if np.isposinf(snr_db_value):
        if preserve_dtype:
            return x.copy()
        return x.astype(output_dtype, copy=True)
    if preserve_dtype and _is_quantizing_dtype(x.dtype):
        raise ValueError(
            "preserve_dtype=True with integer/bool x_t would quantize AWGN. "
            "Use preserve_dtype=False or cast x_t to float/complex first."
        )
    if preserve_dtype and np.dtype(x.dtype) == np.float16:
        warnings.warn(
            "preserve_dtype=True with float16 can heavily quantize AWGN; "
            "consider preserve_dtype=False for better noise fidelity.",
            category=RuntimeWarning,
            stacklevel=2,
        )

    # Estimate average signal power with the standard discrete-time definition.
    signal_power = _compute_signal_power(x)
    if signal_power == 0.0:
        raise ValueError("x_t has zero power; SNR is undefined for a null signal.")

    # Convert SNR from decibels and validate derived AWGN parameters.
    noise_std = _compute_noise_std(signal_power=signal_power, snr_db=snr_db_value)

    # If derived noise is exactly zero, return without consuming RNG state.
    if noise_std == 0.0:
        if np.dtype(x.dtype) == output_dtype:
            return x.copy()
        return x.astype(output_dtype, copy=True)
    rng_obj = rng if rng is not None else np.random.default_rng()

    # Draw AWGN directly in the final output dtype.
    noise = _draw_white_noise(
        shape=x.shape,
        noise_std=noise_std,
        is_complex=is_complex,
        rng=rng_obj,
        output_dtype=output_dtype,
    )

    # Avoid a redundant cast when input and output dtypes already match.
    if np.dtype(x.dtype) == output_dtype:
        return x + noise
    return x.astype(output_dtype, copy=False) + noise


__all__ = ["awgn"]
