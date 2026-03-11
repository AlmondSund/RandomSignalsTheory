"""Tests for Gaussian-bell noise models."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from common.gaussian_bell_noise import gaussian_bell_noise
from common.gaussian_bell_noise import gaussian_bell_psd
from common.gaussian_bell_noise import gaussian_variance_awgn
from common.gaussian_bell_noise import gaussian_variance_profile


def _load_reference_pyc_module() -> object:
    """Loads the cached bytecode module used as a reconstruction reference."""
    pyc_path = Path("common/__pycache__/gaussian_bell_noise.cpython-313.pyc")
    if not pyc_path.exists():
        raise FileNotFoundError(
            "Reference pyc artifact is not available in this worktree."
        )

    loader = importlib.machinery.SourcelessFileLoader(
        "gaussian_bell_noise_reference", str(pyc_path)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(
            "Failed to build an import specification for the reference pyc module."
        )

    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class GaussianBellNoiseTests(unittest.TestCase):
    """Validates the Gaussian-bell PSD and heteroscedastic AWGN helpers."""

    def test_gaussian_bell_psd_matches_closed_form(self) -> None:
        """The PSD helper should evaluate the documented Gaussian formula exactly."""
        frequency_hz = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        psd = gaussian_bell_psd(
            frequency_hz=frequency_hz, spectral_std_hz=0.5, peak_power=3.0
        )
        expected_psd = 3.0 * np.exp(-0.5 * (frequency_hz / 0.5) ** 2)
        np.testing.assert_allclose(psd, expected_psd)
        np.testing.assert_allclose(psd, psd[::-1])

    def test_gaussian_bell_psd_rejects_nonfinite_frequency(self) -> None:
        """The PSD helper should fail fast when the frequency grid is invalid."""
        with self.assertRaisesRegex(
            ValueError, "frequency_hz must contain only finite values"
        ):
            gaussian_bell_psd(
                frequency_hz=np.array([0.0, np.nan], dtype=np.float64),
                spectral_std_hz=1.0,
            )

    def test_gaussian_bell_noise_is_zero_mean_reproducible_and_normalized(self) -> None:
        """The Gaussian-bell colored noise should honor seed and target std."""
        signal_a = gaussian_bell_noise(
            num_samples=256,
            sample_rate_hz=10.0,
            target_std=1.7,
            spectral_std_hz=0.9,
            rng=np.random.default_rng(1234),
        )
        signal_b = gaussian_bell_noise(
            num_samples=256,
            sample_rate_hz=10.0,
            target_std=1.7,
            spectral_std_hz=0.9,
            rng=np.random.default_rng(1234),
        )

        np.testing.assert_allclose(signal_a, signal_b)
        self.assertEqual(signal_a.shape, (256,))
        self.assertLess(abs(float(np.mean(signal_a))), 1e-12)
        self.assertTrue(np.isclose(np.std(signal_a, dtype=np.float64), 1.7, atol=1e-12))

    def test_gaussian_bell_noise_concentrates_power_near_dc(self) -> None:
        """A narrow Gaussian PSD should emphasize low-frequency bins over high ones."""
        rng = np.random.default_rng(77)
        num_realizations = 64
        num_samples = 512

        low_band_power = []
        high_band_power = []
        for _ in range(num_realizations):
            realization = gaussian_bell_noise(
                num_samples=num_samples,
                sample_rate_hz=1.0,
                target_std=1.0,
                spectral_std_hz=0.05,
                rng=rng,
            )
            periodogram = np.abs(np.fft.fft(realization)) ** 2
            abs_frequency_hz = np.abs(np.fft.fftfreq(num_samples, d=1.0))
            low_band_power.append(float(periodogram[abs_frequency_hz <= 0.05].mean()))
            high_band_power.append(float(periodogram[abs_frequency_hz >= 0.25].mean()))

        self.assertGreater(np.mean(low_band_power), 4.0 * np.mean(high_band_power))

    def test_gaussian_variance_profile_matches_closed_form(self) -> None:
        """The variance profile should equal the documented Gaussian bell formula."""
        profile = gaussian_variance_profile(
            num_samples=5,
            profile_std_samples=1.0,
            center_sample=2.0,
            peak_variance=4.0,
        )
        expected_profile = 4.0 * np.exp(
            -0.5 * (np.arange(5, dtype=np.float64) - 2.0) ** 2
        )
        np.testing.assert_allclose(profile, expected_profile)

    def test_gaussian_variance_awgn_tracks_the_requested_variance_envelope(
        self,
    ) -> None:
        """Empirical per-sample variance should follow the Gaussian envelope shape."""
        num_samples = 101
        num_realizations = 512
        rng = np.random.default_rng(2026)

        realizations = np.stack(
            [
                gaussian_variance_awgn(
                    num_samples=num_samples,
                    target_std=2.5,
                    profile_std_samples=12.0,
                    center_sample=50.0,
                    rng=rng,
                )
                for _ in range(num_realizations)
            ],
            axis=0,
        )
        empirical_variance_profile = np.var(realizations, axis=0, ddof=0)
        target_profile = gaussian_variance_profile(
            num_samples=num_samples,
            profile_std_samples=12.0,
            center_sample=50.0,
            peak_variance=1.0,
        )

        normalized_empirical = (
            empirical_variance_profile / empirical_variance_profile.max()
        )
        normalized_target = target_profile / target_profile.max()
        correlation = float(np.corrcoef(normalized_empirical, normalized_target)[0, 1])

        self.assertTrue(
            np.isclose(np.std(realizations[0], dtype=np.float64), 2.5, atol=1e-12)
        )
        self.assertGreater(correlation, 0.98)

    def test_reconstructed_module_matches_cached_reference_bytecode(self) -> None:
        """The source reconstruction should match the cached bytecode behavior."""
        try:
            reference = _load_reference_pyc_module()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

        frequency_hz = np.array([-1.5, -0.25, 0.0, 0.25, 1.5], dtype=np.float64)
        np.testing.assert_allclose(
            gaussian_bell_psd(
                frequency_hz=frequency_hz, spectral_std_hz=0.4, peak_power=1.8
            ),
            reference.gaussian_bell_psd(
                frequency_hz=frequency_hz, spectral_std_hz=0.4, peak_power=1.8
            ),
        )

        np.testing.assert_allclose(
            gaussian_bell_noise(
                num_samples=64,
                sample_rate_hz=4.0,
                target_std=1.25,
                spectral_std_hz=0.35,
                rng=np.random.default_rng(5),
            ),
            reference.gaussian_bell_noise(
                num_samples=64,
                sample_rate_hz=4.0,
                target_std=1.25,
                spectral_std_hz=0.35,
                rng=np.random.default_rng(5),
            ),
        )

        np.testing.assert_allclose(
            gaussian_variance_profile(
                num_samples=9,
                profile_std_samples=2.0,
                center_sample=4.0,
                peak_variance=1.5,
            ),
            reference.gaussian_variance_profile(
                num_samples=9,
                profile_std_samples=2.0,
                center_sample=4.0,
                peak_variance=1.5,
            ),
        )

        np.testing.assert_allclose(
            gaussian_variance_awgn(
                num_samples=64,
                target_std=0.8,
                profile_std_samples=10.0,
                center_sample=20.5,
                rng=np.random.default_rng(7),
            ),
            reference.gaussian_variance_awgn(
                num_samples=64,
                target_std=0.8,
                profile_std_samples=10.0,
                center_sample=20.5,
                rng=np.random.default_rng(7),
            ),
        )


if __name__ == "__main__":
    unittest.main()
