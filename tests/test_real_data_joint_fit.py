from pathlib import Path

import numpy as np
import pytest

from astromodels import Model, PointSource
from astromodels.core.polarization import LinearPolarization
from astromodels.core.spectral_component import SpectralComponent
from astromodels.functions.functions_1D.powerlaws import Band
from threeML import DataList
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder

# Define path to the real test data directory relative to this test file
TEST_DATA_DIR = Path(__file__).parent / "data"

TRIGGER_TIME = 1474786761.0


@pytest.fixture
def real_data_paths():
    """Paths to the real POLAR GRB160325A files in tests/data/."""
    return {
        "polar_pevt": TEST_DATA_DIR / "POLAR_160325A.pevt",
        "polar_prsp": TEST_DATA_DIR / "POLAR_160325A.prsp",
    }


def _build_polar_plugin(real_data_paths):
    ts = TimeSeriesBuilder.from_polarization(
        "polar_pol",
        str(real_data_paths["polar_pevt"]),
        str(real_data_paths["polar_prsp"]),
        trigger_time=TRIGGER_TIME,
    )
    ts.set_background_interval("-300.0--0.0", "50-300")
    ts.set_active_time_interval("0.5-40.0")
    return ts.to_polarizationlike()


def test_real_data_loading_and_datalist_building(real_data_paths):
    """Tests that real .pevt and .prsp files can be parsed by TimeSeriesBuilder,
    converted to a PolarizationLike plugin, and combined into a 3ML DataList."""
    if not real_data_paths["polar_pevt"].exists() or not real_data_paths["polar_prsp"].exists():
        pytest.skip("Real sample data files not found in tests/data/. Skipping real data test.")

    plugin = _build_polar_plugin(real_data_paths)
    datalist = DataList(plugin)

    assert list(datalist.keys()) == ["polar_pol"]


def test_real_data_joint_likelihood_evaluation(real_data_paths):
    """Tests full model setup (Band spectrum + LinearPolarization) against real
    data and evaluates the plugin's log-likelihood to ensure spectral +
    polarization interpolation executes cleanly without NaNs."""
    if not real_data_paths["polar_pevt"].exists() or not real_data_paths["polar_prsp"].exists():
        pytest.skip("Real sample data files not found in tests/data/. Skipping real data test.")

    plugin = _build_polar_plugin(real_data_paths)

    # Build spectral + polarization model (published GRB160325A values)
    band = Band()
    band.xp.value = 266.04
    band.K.value = 0.17
    band.alpha.value = -0.77
    band.beta.value = -2.67

    lp = LinearPolarization(degree=50.0, angle=156.0)

    sc = SpectralComponent("synch", band, lp)
    ps = PointSource("GRB160325A", 0, 0, components=[sc])
    model = Model(ps)

    plugin.set_model(model)
    log_like = plugin.get_log_like()

    assert np.isfinite(log_like), f"Computed log-likelihood is not finite: {log_like}"
    assert isinstance(log_like, float)
