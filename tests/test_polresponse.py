import numpy as np

from polpy.polresponse import PolResponse, harmonic


def test_harmonic_function():
    """Verify standard response harmonic function behavior and periodicity."""
    x = np.array([0.0, 90.0, 180.0, 360.0])
    # Evaluating harmonic: const + ampl1*sin(x + phi1) + ampl2*sin(2x + phi2)
    res = harmonic(x, const=10.0, ampl1=2.0, phi1=0.0, ampl2=1.0, phi2=0.0)

    assert res.shape == (4,)
    # Periodicity check (0 deg vs 360 deg)
    assert np.isclose(res[0], res[3])


def test_pol_response_interpolation(mock_prsp_file):
    """Verify PolResponse matrix sorting and grid point evaluation."""
    pa_offset = 15.0
    rsp = PolResponse(mock_prsp_file, pa_offset=pa_offset)

    assert rsp.n_scattering_bins == 2
    assert len(rsp.ene_center) == 2

    # Evaluate grid point for 50% polarization fraction and 45 deg angle
    grid_eval = rsp.evaluate_grid_point(pol_ang_val=45.0, pol_deg_val=50.0)

    # Expected output shape: (N_Energies, N_Scattering_Bins) -> (2, 2)
    assert grid_eval.shape == (2, 2)
    assert np.all(np.isfinite(grid_eval))


def test_pol_response_angle_sorted(mock_prsp_file):
    """The response's pol_ang array must come out sorted ascending, since
    evaluate_grid_point/harmonic fitting assumes a sorted grid."""
    rsp = PolResponse(mock_prsp_file, pa_offset=0.0)

    assert np.all(np.diff(rsp.pol_ang) >= 0)


def test_pol_response_boundary_polarization(mock_prsp_file):
    """Test PF = 0% (pure unpolarized matrix) and PF = 100% (pure polarized
    matrix) evaluation at the boundaries."""
    rsp = PolResponse(mock_prsp_file, pa_offset=0.0)

    fully_unpolarized = rsp.evaluate_grid_point(pol_ang_val=45.0, pol_deg_val=0.0)
    assert np.allclose(fully_unpolarized, rsp.unpol_matrix)

    fully_polarized = rsp.evaluate_grid_point(pol_ang_val=45.0, pol_deg_val=100.0)
    expected = harmonic(45.0, rsp.const, rsp.ampl1, rsp.phi1, rsp.ampl2, rsp.phi2)
    assert np.allclose(fully_polarized, expected)


def test_pol_response_negative_pa_offset_wraps(mock_prsp_file):
    """Ensure (180 + pol_ang - pa_offset) % 180 handles negative offsets/inputs
    correctly, i.e. all resulting angles stay within [0, 180)."""
    rsp = PolResponse(mock_prsp_file, pa_offset=-370.0)

    assert np.all(rsp.pol_ang >= 0.0)
    assert np.all(rsp.pol_ang < 180.0)
