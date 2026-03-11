"""Gaussian-shaped noise models in frequency and time."""

from __future__ import annotations

import numpy as np


def _validate_positive_scalar(
    value: float,  # Candidate scalar parameter
    name: str,  # Parameter name used in error messages
) -> float:  # Validated strictly positive scalar
    """Validates that a parameter is a finite, strictly positive scalar."""
    try:
        scalar_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar.") from exc

    if not np.isfinite(scalar_value) or scalar_value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")

    return scalar_value


def _validate_nonnegative_scalar(
    value: float,  # Candidate scalar parameter
    name: str,  # Parameter name used in error messages
) -> float:  # Validated non-negative scalar
    """Validates that a parameter is a finite, non-negative scalar."""
    try:
        scalar_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar.") from exc

    if not np.isfinite(scalar_value) or scalar_value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")

    return scalar_value


def gaussian_bell_psd(
    frequency_hz: np.ndarray,  # Two-sided frequency grid [Hz]
    spectral_std_hz: float,  # Gaussian PSD standard deviation [Hz]
    peak_power: float = 1.0,  # PSD value at 0 Hz [a.u./Hz]
) -> np.ndarray:  # Gaussian-bell PSD samples
    """Evaluates a zero-centered Gaussian-bell PSD on a frequency grid.

    Purpose
    -------
    Produces a non-negative, even spectral envelope that can be used as the
    target PSD for a colored Gaussian noise model.

    Parameters
    ----------
    frequency_hz : np.ndarray
        Two-sided frequency grid [Hz]. The function accepts any NumPy-shaped
        array and returns a PSD with the same shape.
    spectral_std_hz : float
        Standard deviation of the Gaussian bell [Hz]. Smaller values
        concentrate power more tightly around 0 Hz.
    peak_power : float, default=1.0
        Peak PSD value at 0 Hz. This is a relative scaling factor unless the
        caller calibrates it to physical PSD units.

    Returns
    -------
    np.ndarray
        Non-negative PSD samples with the same shape as ``frequency_hz``.

    Side Effects
    ------------
    None.
    """
    spectral_std_hz_value = _validate_positive_scalar(
        spectral_std_hz, "spectral_std_hz"
    )
    peak_power_value = _validate_nonnegative_scalar(peak_power, "peak_power")

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must contain only finite values.")

    normalized_frequency = frequency / spectral_std_hz_value
    return peak_power_value * np.exp(-0.5 * normalized_frequency * normalized_frequency)


def gaussian_bell_noise(
    num_samples: int,  # Number of output samples
    sample_rate_hz: float = 1.0,  # Sampling frequency used for the FFT grid [Hz]
    target_std: float = 1.0,  # Requested time-domain standard deviation [a.u.]
    spectral_std_hz: float = 0.1,  # Gaussian PSD standard deviation [Hz]
    rng: np.random.Generator | None = None,  # RNG for reproducible synthesis
) -> np.ndarray:  # Real-valued colored Gaussian noise
    """Generates Gaussian noise whose PSD follows a Gaussian bell.

    Purpose
    -------
    Draws a real-valued Gaussian process by shaping white Gaussian noise in the
    frequency domain with the square root of a Gaussian target PSD.

    Parameters
    ----------
    num_samples : int
        Number of output samples.
    sample_rate_hz : float, default=1.0
        Sampling frequency [Hz]. It defines the FFT-bin frequency axis used for
        PSD shaping.
    target_std : float, default=1.0
        Desired time-domain standard deviation of the returned realization.
        ``target_std = 0`` returns an all-zero sequence.
    spectral_std_hz : float, default=0.1
        Standard deviation of the Gaussian PSD [Hz]. Smaller values produce a
        narrower low-pass spectrum.
    rng : np.random.Generator | None, default=None
        Random generator used for reproducible noise draws.

    Returns
    -------
    np.ndarray
        Real-valued Gaussian noise with shape ``(num_samples,)``.

    Side Effects
    ------------
    Consumes random numbers from ``rng`` or from a newly created generator.

    Assumptions
    -----------
    - The PSD is centered at 0 Hz and therefore favors low-frequency content.
    - The returned realization is normalized to ``target_std`` after synthesis,
      so finite-length draws match the requested variance exactly up to floating
      point rounding.
    """
    if not isinstance(num_samples, int):
        raise ValueError("num_samples must be an integer.")
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2 to define a meaningful PSD shape.")

    sample_rate_hz_value = _validate_positive_scalar(sample_rate_hz, "sample_rate_hz")
    target_std_value = _validate_nonnegative_scalar(target_std, "target_std")
    spectral_std_hz_value = _validate_positive_scalar(
        spectral_std_hz, "spectral_std_hz"
    )

    if target_std_value == 0.0:
        return np.zeros(num_samples, dtype=np.float64)

    rng_obj = rng if rng is not None else np.random.default_rng()

    # Build the two-sided FFT frequency grid so the target PSD is explicit.
    frequency_hz = np.fft.fftfreq(num_samples, d=1.0 / sample_rate_hz_value)
    shaping_psd = gaussian_bell_psd(
        frequency_hz=frequency_hz,
        spectral_std_hz=spectral_std_hz_value,
        peak_power=1.0,
    )

    # Shape white Gaussian noise by multiplying its spectrum by sqrt(PSD).
    white_noise = rng_obj.standard_normal(num_samples)
    white_spectrum = np.fft.fft(white_noise)
    shaped_spectrum = white_spectrum * np.sqrt(shaping_psd)

    # Return a centered, real-valued realization with the requested variance.
    colored_noise = np.fft.ifft(shaped_spectrum).real
    colored_noise -= np.mean(colored_noise, dtype=np.float64)

    empirical_std = float(np.std(colored_noise, dtype=np.float64))
    if empirical_std == 0.0:
        raise RuntimeError(
            "Generated a degenerate realization with zero variance; increase num_samples or spectral_std_hz."
        )

    return colored_noise * (target_std_value / empirical_std)


def gaussian_variance_profile(
    num_samples: int,  # Number of samples in the profile
    profile_std_samples: float,  # Gaussian profile standard deviation [samples]
    center_sample: float | None = None,  # Bell center index [samples]
    peak_variance: float = 1.0,  # Variance at the bell center [a.u.^2]
) -> np.ndarray:  # Deterministic Gaussian variance envelope
    """Evaluates a Gaussian variance envelope over discrete time.

    Purpose
    -------
    Produces a deterministic time-varying variance profile that can be used to
    synthesize heteroscedastic AWGN whose power is concentrated around one time
    region rather than remaining constant across the record.

    Parameters
    ----------
    num_samples : int
        Number of time samples in the profile.
    profile_std_samples : float
        Standard deviation of the Gaussian variance bell [samples].
    center_sample : float | None, default=None
        Center of the Gaussian bell [samples]. If omitted, the profile is
        centered in the middle of the record.
    peak_variance : float, default=1.0
        Maximum variance at the bell center [a.u.^2].

    Returns
    -------
    np.ndarray
        Non-negative variance profile with shape ``(num_samples,)``.

    Side Effects
    ------------
    None.
    """
    if not isinstance(num_samples, int):
        raise ValueError("num_samples must be an integer.")
    if num_samples < 2:
        raise ValueError(
            "num_samples must be >= 2 to define a meaningful variance profile."
        )

    profile_std_samples_value = _validate_positive_scalar(
        profile_std_samples, "profile_std_samples"
    )
    peak_variance_value = _validate_nonnegative_scalar(peak_variance, "peak_variance")

    if center_sample is None:
        center_sample_value = 0.5 * (num_samples - 1)
    else:
        center_sample_value = _validate_nonnegative_scalar(
            center_sample, "center_sample"
        )
        if center_sample_value > num_samples - 1:
            raise ValueError(
                "center_sample must lie within the available sample range."
            )

    sample_index = np.arange(num_samples, dtype=np.float64)
    normalized_time = (sample_index - center_sample_value) / profile_std_samples_value
    return peak_variance_value * np.exp(-0.5 * normalized_time * normalized_time)


def gaussian_variance_awgn(
    num_samples: int,  # Number of output samples
    target_std: float = 1.0,  # Requested global standard deviation [a.u.]
    profile_std_samples: float = 32.0,  # Gaussian variance profile standard deviation [samples]
    center_sample: float | None = None,  # Variance-bell center [samples]
    rng: np.random.Generator | None = None,  # RNG for reproducible synthesis
) -> np.ndarray:  # Heteroscedastic real Gaussian noise
    """Generates AWGN with a Gaussian time-varying variance envelope.

    Purpose
    -------
    Draws a real-valued Gaussian sequence whose samples are independent but not
    identically distributed: the local variance follows a Gaussian bell over
    time. This implements a heteroscedastic additive noise model.

    Parameters
    ----------
    num_samples : int
        Number of output samples.
    target_std : float, default=1.0
        Desired global standard deviation of the realization [a.u.].
        ``target_std = 0`` returns an all-zero sequence.
    profile_std_samples : float, default=32.0
        Standard deviation of the Gaussian variance bell [samples].
    center_sample : float | None, default=None
        Center of the Gaussian variance bell [samples]. If omitted, the bell is
        centered in the middle of the record.
    rng : np.random.Generator | None, default=None
        Random generator used for reproducible draws.

    Returns
    -------
    np.ndarray
        Real-valued heteroscedastic Gaussian noise with shape
        ``(num_samples,)``.

    Side Effects
    ------------
    Consumes random numbers from ``rng`` or from a newly created generator.

    Assumptions
    -----------
    - The process is white only in the conditional sense of independent
      Gaussian samples; it is not wide-sense stationary because its variance
      depends on time.
    - The average power of the returned realization is normalized to
      ``target_std**2`` up to floating-point rounding.
    """
    target_std_value = _validate_nonnegative_scalar(target_std, "target_std")
    if target_std_value == 0.0:
        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be an integer.")
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0.")
        return np.zeros(num_samples, dtype=np.float64)

    variance_profile = gaussian_variance_profile(
        num_samples=num_samples,
        profile_std_samples=profile_std_samples,
        center_sample=center_sample,
        peak_variance=1.0,
    )

    rng_obj = rng if rng is not None else np.random.default_rng()

    # Apply the square root of the variance envelope to white innovations.
    white_noise = rng_obj.standard_normal(num_samples)
    shaped_noise = white_noise * np.sqrt(variance_profile)
    shaped_noise -= np.mean(shaped_noise, dtype=np.float64)

    empirical_std = float(np.std(shaped_noise, dtype=np.float64))
    if empirical_std == 0.0:
        raise RuntimeError(
            "Generated a degenerate realization with zero variance; increase num_samples or profile_std_samples."
        )

    return shaped_noise * (target_std_value / empirical_std)


__all__ = [
    "gaussian_bell_psd",
    "gaussian_bell_noise",
    "gaussian_variance_profile",
    "gaussian_variance_awgn",
]
