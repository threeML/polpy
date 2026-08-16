import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from astromodels import Model, PointSource
from astromodels.core.polarization import LinearPolarization
from astromodels.functions import Powerlaw

from polpy.polarizationlike import PolarizationLike
from polpy.polresponse import PolResponse


class MockObservation:
    def __init__(self, n_bins=2):
        self.counts = np.array([100, 120])
        self.exposure = 10.0
        self.edges = [0.0, 180.0, 360.0]


class MockBackground:
    def __init__(self, n_bins=2):
        self.counts = np.array([20, 25])
        self.count_errors = np.array([4.4, 5.0])
        self.exposure = 10.0


def test_polarizationlike_init_and_nuisance(mock_prsp_file):
    """Test PolarizationLike setup and effective area correction toggles."""
    obs = MockObservation()
    bkg = MockBackground()
    rsp = PolResponse(mock_prsp_file, pa_offset=0.0)

    plugin = PolarizationLike("test_pol", obs, bkg, rsp)

    # Check effective area nuisance parameter defaults
    assert plugin.effective_area_correction.value == 1.0
    assert not plugin.effective_area_correction.free

    # Enable area correction
    plugin.use_effective_area_correction(lower=0.7, upper=1.3)
    assert plugin.effective_area_correction.free
    assert plugin.effective_area_correction.bounds == (0.7, 1.3)

    # Fix area correction
    plugin.fix_effective_area_correction(value=1.1)
    assert not plugin.effective_area_correction.free
    assert plugin.effective_area_correction.value == 1.1


def test_polarizationlike_shape_mismatch_raises(mock_prsp_file):
    """Observation counts length must match the response's scattering bins."""

    class BadObservation(MockObservation):
        def __init__(self):
            super().__init__()
            self.counts = np.array([100, 120, 130])  # 3 bins, response has 2

    obs = BadObservation()
    bkg = MockBackground()
    rsp = PolResponse(mock_prsp_file, pa_offset=0.0)

    with pytest.raises(AssertionError):
        PolarizationLike("test_pol", obs, bkg, rsp)


def test_polarizationlike_model_setting_and_likelihood(mock_prsp_file):
    """Test Astromodels parameter binding and Poisson log-likelihood calculation."""
    obs = MockObservation()
    bkg = MockBackground()
    rsp = PolResponse(mock_prsp_file, pa_offset=0.0)
    plugin = PolarizationLike("test_pol", obs, bkg, rsp)

    # Construct Astromodels model with polarization
    po = Powerlaw()
    pol = LinearPolarization(degree=30.0, angle=45.0)
    src = PointSource("src", 0.0, 0.0, spectral_shape=po, polarization=pol)
    model = Model(src)

    plugin.set_model(model)

    # Ensure parameter extraction bound correctly
    assert plugin._pol_degree.value == 30.0
    assert plugin._pol_angle.value == 45.0

    # Compute log-likelihood
    log_like = plugin.get_log_like()
    assert isinstance(log_like, float)
    assert np.isfinite(log_like)


def test_polarizationlike_nan_safety_zero_counts(mock_prsp_file):
    """get_log_like() should stay finite via np.nansum even with zero observed
    counts in a bin (log(0/model) term would otherwise be -inf)."""

    class ZeroObservation(MockObservation):
        def __init__(self):
            super().__init__()
            self.counts = np.array([0, 120])

    obs = ZeroObservation()
    bkg = MockBackground()
    rsp = PolResponse(mock_prsp_file, pa_offset=0.0)
    plugin = PolarizationLike("test_pol", obs, bkg, rsp)

    po = Powerlaw()
    pol = LinearPolarization(degree=10.0, angle=20.0)
    src = PointSource("src", 0.0, 0.0, spectral_shape=po, polarization=pol)
    plugin.set_model(Model(src))

    log_like = plugin.get_log_like()
    assert np.isfinite(log_like)


def test_polarizationlike_display(mock_prsp_file):
    """Verify that display() returns a valid Matplotlib Figure object."""
    obs = MockObservation()
    bkg = MockBackground()
    rsp = PolResponse(mock_prsp_file, pa_offset=0.0)
    plugin = PolarizationLike("test_pol", obs, bkg, rsp)

    po = Powerlaw()
    pol = LinearPolarization(degree=10.0, angle=20.0)
    src = PointSource("src", 0.0, 0.0, spectral_shape=po, polarization=pol)
    plugin.set_model(Model(src))

    fig = plugin.display()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
