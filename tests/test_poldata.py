import numpy as np

from polpy.poldata import PolData


def test_poldata_initialization_and_masking(mock_pevt_file):
    """Test header parsing and masking of bad PHA and scattering angles."""
    data = PolData(mock_pevt_file, reference_time=0.0)

    assert data.mission == "TEST_TELESCOPE"
    assert data.instrument == "POLARIMETER"
    assert data.n_channels == 10
    assert data.n_scattering_bins == 12

    # Check PHA masking (CHANNEL >= 0)
    # Original CHANNEL had 5 items, one was -1
    assert len(data.pha) == 4
    assert np.all(data.pha >= 0)

    # Check total masking (CHANNEL >= 0 AND SABIN != -1)
    # Item at index 2 had bad channel, item at index 3 had bad SABIN (-1)
    assert len(data.scattering_angles) == 3
    assert np.all(data.scattering_angles != -1)


def test_poldata_pa_offset_calculation(mock_pevt_file):
    """Test polarization angle offset calculation returns a finite floating point value."""
    data = PolData(mock_pevt_file)
    pa_offset = data.get_pa_offset()

    assert isinstance(pa_offset, float)
    assert np.isfinite(pa_offset)
    assert -180.0 <= pa_offset <= 180.0


def test_poldata_pa_offset_southern_hemisphere(mock_pevt_file, tmp_path):
    """get_pa_offset() should also return a sane value for sources with Dec < 0."""
    import shutil
    from astropy.io import fits

    # Reuse the mock file but flip the source declination to the southern sky
    south_path = tmp_path / "mock_events_south.pevt"
    shutil.copy(mock_pevt_file, south_path)

    with fits.open(south_path, mode="update") as hdul:
        hdul["POLEVENTS"].header["DECGRB"] = -30.0

    data = PolData(str(south_path))
    pa_offset = data.get_pa_offset()

    assert isinstance(pa_offset, float)
    assert np.isfinite(pa_offset)
    assert -180.0 <= pa_offset <= 180.0
