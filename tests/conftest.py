import sys
import types

import numpy as np
import pytest
from astropy.io import fits


def _ensure_importable_pymultinest():
    """threeML unconditionally does `import pymultinest` at package-import time
    to discover available Bayesian samplers. On systems without the compiled
    MultiNest library (e.g. Windows, or a plain `pip install` CI image without
    libmultinest), pymultinest's loader calls sys.exit(1) instead of raising
    ImportError, which aborts the whole pytest run before collection even
    starts. None of these tests exercise the MultiNest sampler itself, so if
    the real import fails this way, swap in an inert stub module so threeML's
    sampler-discovery loop can move on.
    """
    try:
        import pymultinest  # noqa: F401
    except SystemExit:
        stub = types.ModuleType("pymultinest")
        stub.run = None
        stub.analyse = types.ModuleType("pymultinest.analyse")
        sys.modules["pymultinest"] = stub
        sys.modules["pymultinest.analyse"] = stub.analyse


_ensure_importable_pymultinest()

# threeML.utils.data_builders.time_series_builder detects PolPy availability via
# `from polpy.polarizationlike import PolarizationLike` at threeML-import time.
# polpy.polarizationlike in turn does `from threeML import PluginPrototype` at
# polpy-import time. If a test imports `polpy.polarizationlike` before threeML
# has ever been imported in the process, this becomes a circular import:
# threeML's own probe re-enters the not-yet-finished polpy.polarizationlike
# module and fails, silently leaving threeML's `has_polpy = False` and breaking
# TimeSeriesBuilder.from_polarization/to_polarizationlike for the whole session.
# Importing threeML first, once, here avoids that ordering hazard.
import threeML  # noqa: F401, E402

from polpy.polresponse import harmonic  # noqa: E402


@pytest.fixture
def mock_pevt_file(tmp_path):
    """Generates a synthetic POLEVENTS FITS file for testing PolData."""
    file_path = tmp_path / "mock_events.pevt"

    primary_hdu = fits.PrimaryHDU()

    hdr = fits.Header()
    hdr["TELESCOP"] = "TEST_TELESCOPE"
    hdr["INSTRUME"] = "POLARIMETER"
    hdr["NCHANS"] = 10
    hdr["NSABINS"] = 12
    hdr["RAX"] = 0.0
    hdr["DECX"] = 0.0
    hdr["RAZ"] = 0.0
    hdr["DECZ"] = 90.0
    hdr["RAGRB"] = 45.0
    hdr["DECGRB"] = 30.0

    cols = [
        fits.Column(name="CHANNEL", format="I", array=np.array([1, 2, -1, 4, 5])),
        fits.Column(name="SABIN", format="I", array=np.array([0, 5, 2, -1, 10])),
        fits.Column(
            name="DEADFRAC", format="E", array=np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        ),
        fits.Column(
            name="TIME", format="D", array=np.array([10.0, 10.5, 11.0, 11.5, 12.0])
        ),
    ]
    evt_hdu = fits.BinTableHDU.from_columns(cols, header=hdr, name="POLEVENTS")

    hdul = fits.HDUList([primary_hdu, evt_hdu])
    hdul.writeto(file_path)
    return str(file_path)


@pytest.fixture
def mock_prsp_file(tmp_path):
    """Generates a synthetic .prsp response file for PolResponse."""
    file_path = tmp_path / "mock_response.prsp"

    primary_hdu = fits.PrimaryHDU()

    # INEBOUNDS extension
    ine_cols = [
        fits.Column(name="ENERG_LO", format="E", array=np.array([10.0, 50.0])),
        fits.Column(name="ENERG_HI", format="E", array=np.array([50.0, 100.0])),
    ]
    ine_hdu = fits.BinTableHDU.from_columns(ine_cols, name="INEBOUNDS")

    # SABOUNDS extension
    sa_cols = [
        fits.Column(name="SA_MIN", format="E", array=np.array([0.0, 180.0])),
        fits.Column(name="SA_MAX", format="E", array=np.array([180.0, 360.0])),
    ]
    sa_hdu = fits.BinTableHDU.from_columns(sa_cols, name="SABOUNDS")

    # INPAVALS extension. _interpolate_rsp() fits a 5-parameter harmonic model
    # per (energy, scattering-bin) cell via curve_fit, which requires at least
    # 5 PA points to not be underdetermined.
    pa_in = np.array([0.0, 36.0, 72.0, 108.0, 144.0])
    pa_cols = [
        fits.Column(name="PA_IN", format="E", array=pa_in),
    ]
    pa_hdu = fits.BinTableHDU.from_columns(pa_cols, name="INPAVALS")

    # SPECRESP POLMATRIX (N_SA=2, N_PA=5, N_E=2) -- PolResponse.__init__ transposes
    # this to (N_E, N_PA, N_SA). Values follow the harmonic model itself (with a
    # touch of per-PA variation) so the least-squares fit has a real, non-degenerate
    # solution rather than fitting noise.
    _true_harmonic = harmonic(
        pa_in, const=100.0, ampl1=10.0, phi1=0.3, ampl2=5.0, phi2=1.1
    )
    pol_matrix_data = np.stack(
        [
            np.stack([_true_harmonic, _true_harmonic + 20.0], axis=-1),
            np.stack([_true_harmonic * 0.95, (_true_harmonic + 20.0) * 0.95], axis=-1),
        ],
        axis=0,
    ).astype(np.float32)
    pol_matrix_hdu = fits.ImageHDU(pol_matrix_data, name="SPECRESP POLMATRIX")

    # SPECRESP UNPOLMATRIX (N_SA=2, N_E=2) -- transposed to (N_E, N_SA)
    unpol_matrix_data = np.ones((2, 2), dtype=np.float32) * 50.0
    unpol_matrix_hdu = fits.ImageHDU(unpol_matrix_data, name="SPECRESP UNPOLMATRIX")

    hdul = fits.HDUList(
        [primary_hdu, ine_hdu, sa_hdu, pa_hdu, pol_matrix_hdu, unpol_matrix_hdu]
    )
    hdul.writeto(file_path)
    return str(file_path)
