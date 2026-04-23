import numba as nb
from numba.experimental import jitclass
import numpy as np
from interpolation.splines import eval_linear
from astropy.io import fits

#import scipy.interpolate as interpolate


spec = [
    ("_values", nb.float64[:, :, :]),
    (
        "_grid",
        nb.typeof(
            (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        ),
    ),
]


@jitclass(spec)
class FastGridInterpolate(object):

    def __init__(self, grid, values):
        self._grid = grid
        self._values = np.ascontiguousarray(values)

    def evaluate(self, v):

        return eval_linear(self._grid, self._values, v)


class PolResponse(object):

    def __init__(self, response_file, pa_offset, interp_method='linear', n_harmonics=2, fine_pa_resolution=0.01):
        """
        Construct the polarisation response from the mission specific polarisation response file.

        :param response_file: Polarisation response file in the defined format (.prsp)
        :param pa_offset: Offset to be added to convert templates from LTP to J2000
        :param interp_method: Method for interpolating PA ('harmonic' or 'linear'). Defaults to 'harmonic'.
        :param n_harmonics: Number of harmonics to use for the least-squares fit (if interp_method='harmonic').
        :param fine_pa_resolution: Step size in degrees for the high-resolution grid (if interp_method='harmonic').
        """
        print(response_file)
        self._rsp_file = response_file
        self.interp_method = interp_method
        self.n_harmonics = n_harmonics
        self.fine_pa_resolution = fine_pa_resolution

        # read and extract necessary arrays from the response file
        rspHDU = fits.open(self._rsp_file)

        # load energy bouds
        self.ene_lo = rspHDU['INEBOUNDS'].data.field('ENERG_LO')
        self.ene_hi = rspHDU['INEBOUNDS'].data.field('ENERG_HI')

        # energy centre
        self.ene_center = (self.ene_lo + self.ene_hi) / 2.0

        # load input polarization angles
        self.pol_ang = rspHDU['INPAVALS'].data.field('PA_IN')
        self.pol_ang = (180 + self.pol_ang - pa_offset) % 180

        # sort the angles so that we can interpolate correctly
        sorted_indices = np.argsort(self.pol_ang)
        self.pol_ang = self.pol_ang[sorted_indices]

        # read the polmatrix and sort according to the pol angles
        self.pol_matrix = rspHDU['SPECRESP POLMATRIX'].data
        self.pol_matrix = self.pol_matrix[:, sorted_indices, :]
        self.pol_matrix = self.pol_matrix.transpose()

        # read scattering angle bins
        samin = rspHDU['SABOUNDS'].data['SA_MIN']
        samax = rspHDU['SABOUNDS'].data['SA_MAX']
        self.sa_bin = (samin + samax) / 2.0

        # read the unpol matrix
        self.unpol_matrix = rspHDU['SPECRESP UNPOLMATRIX'].data
        self.unpol_matrix = self.unpol_matrix.transpose() # Shape: (N_E, N_SA)

        # pre interpolate the response for fitting

        self._interpolate_rsp()

    def _interpolate_rsp(self):
        """
        Builds the interpolator for the response. This is currently incredibly slow
        and should be improved

        """

        # now go through the response and extract things
        with fits.open(self._rsp_file) as hdu_pol:

            ene_lo = np.array(hdu_pol['INEBOUNDS'].data.field('ENERG_LO'), dtype=np.float64)
            ene_hi = np.array(hdu_pol['INEBOUNDS'].data.field('ENERG_HI'), dtype=np.float64)
            
            energy = (ene_lo + ene_hi) / 2.

            pol_ang = np.array(hdu_pol['INPAVALS'].data.field('PA_IN'), dtype=np.float64)
            pol_ang = (180 + pol_ang - self._pa_offset) % 180
            
            # sort the angles so that we can interpolate correctly
            sorted_indices = np.argsort(pol_ang)
            pol_ang = pol_ang[sorted_indices]

            # we have 100% pol and 0% pol matrix in the prsp file
            pol_deg = np.array([0., 100.], dtype=np.float64)

            samin = np.array(hdu_pol['SABOUNDS'].data.field('SA_MIN'), dtype=np.float64)
            samax = np.array(hdu_pol['SABOUNDS'].data.field('SA_MAX'), dtype=np.float64)
            bins = np.append(samin, samax[-1])
            # get the bin centers as these are where things
            # should be evaluated
            bin_center = 0.5 * (bins[:-1] + bins[1:])

            polmatrix = hdu_pol['SPECRESP POLMATRIX'].data
            
            # we need to sort the polmatrix according to the sorted pol angles
            polmatrix = polmatrix[:, sorted_indices, :]
            polmatrix = polmatrix.transpose()
            # ---------------------------------------------------------
            # NEW: Harmonic Fitting Logic for Polarization Angle
            # ---------------------------------------------------------
            if self.interp_method.lower() == 'harmonic':
                # Create a fine, regular grid from 0 to 180 (inclusive to prevent bounds errors)
                num_fine_points = int(180 / self.fine_pa_resolution) + 1
                fine_grid_angles = np.linspace(0, 180, num_fine_points)
                
                # Convert to radians
                pa_rad = np.deg2rad(pol_ang)
                fine_pa_rad = np.deg2rad(fine_grid_angles)

                # Build design matrices X for the original and fine grid
                X = np.ones((len(pa_rad), 1))
                X_fine = np.ones((len(fine_pa_rad), 1))

                for i in range(1, self.n_harmonics + 1):
                    X = np.hstack([X, np.cos(2 * i * pa_rad[:, None]), np.sin(2 * i * pa_rad[:, None])])
                    X_fine = np.hstack([X_fine, np.cos(2 * i * fine_pa_rad[:, None]), np.sin(2 * i * fine_pa_rad[:, None])])

                N_E, _, N_SA = polmatrix.shape
                polmatrix_fine = np.zeros((N_E, len(fine_grid_angles), N_SA))

                # Loop over energies and compute the least squares fit for all SA bins simultaneously
                for e in range(N_E):
                    y = polmatrix[e, :, :]  # Shape: (N_PA, N_SA)
                    # w = (X^T X)^-1 X^T y. Output w shape is (n_features, N_SA)
                    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    
                    # Compute continuous smooth curve and map to fine grid
                    polmatrix_fine[e, :, :] = X_fine @ w

                # Overwrite variables so the linear interpolator consumes the fine grid
                pol_ang = fine_grid_angles
                polmatrix = polmatrix_fine
            # ---------------------------------------------------------

            uppolmatrix = hdu_pol['SPECRESP UNPOLMATRIX'].data
            uppolmatrix = uppolmatrix.transpose() # Shape: (N_E, N_SA)
            
            # Replicate the unpolarized matrix to match the new PA dimension size
            uppolmatrix = [uppolmatrix] * pol_ang.size
            uppolmatrix = np.stack(uppolmatrix, axis=1) # Shape: (N_E, N_PA, N_SA)

            # Stack into final matrix. Shape: (N_E, N_PA, 2, N_SA)
            pol_matrix = np.stack((uppolmatrix, polmatrix), axis=2)
            pol_matrix = np.array(pol_matrix, dtype=np.float64)

            all_interp = []

            # now we construct a series of interpolation functions that are called during the fit.
            for i, bm in enumerate(bin_center):

                this_interpolator = FastGridInterpolate(
                    (energy, pol_ang, pol_deg), pol_matrix[..., i])

                all_interp.append(this_interpolator)

            # finally we attach all of this to the class
            self.interpolators = all_interp

            self.ene_lo = ene_lo
            self.ene_hi = ene_hi
            self.energy_mid = energy

            self.n_scattering_bins = len(bin_center)
            self.scattering_bins = bin_center
            self.scattering_bins_lo = bins[:-1]
            self.scattering_bins_hi = bins[1:]            


    
