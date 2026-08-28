import numba as nb
import numpy as np
from astropy.io import fits

from scipy.optimize import curve_fit


@nb.njit(fastmath=True)
def harmonic(x, const, ampl1, phi1, ampl2, phi2):
    """Define the 2nd degree harmonic function to interpolate response across PAs for given
    azimuthal bin.

    """

    x = np.deg2rad(x)
    y = const + ampl1 * np.sin(x + phi1) + ampl2 * np.sin(2 * x + phi2)

    return y


class PolResponse(object):

    def __init__(self, response_file, pa_offset):
        """
        Construct the polarisation response from the mission specific polarisation response file.

        :param response_file: Polarisation response file in the defined format (.prsp)
        :param pa_offset: Offset to be subtracted to convert templates from ILTP to IAU
        """
        print(response_file)
        self._rsp_file = response_file

        # read and extract necessary arrays from the response file
        rspHDU = fits.open(self._rsp_file)

        # load energy bouds
        self.ene_lo = rspHDU["INEBOUNDS"].data.field("ENERG_LO")
        self.ene_hi = rspHDU["INEBOUNDS"].data.field("ENERG_HI")

        # energy centre
        self.ene_center = (self.ene_lo + self.ene_hi) / 2.0

        # read scattering angle bins
        samin = rspHDU["SABOUNDS"].data["SA_MIN"]
        samax = rspHDU["SABOUNDS"].data["SA_MAX"]
        self.sa_bin = (samin + samax) / 2.0
        self.n_scattering_bins = self.sa_bin.size

        # load input polarization angles
        self.pol_ang = rspHDU["INPAVALS"].data.field("PA_IN")
        self.pol_ang = (180 + self.pol_ang - pa_offset) % 180

        # sort the angles so that we can interpolate correctly
        sorted_indices = np.argsort(self.pol_ang)
        self.pol_ang = self.pol_ang[sorted_indices]

        # read the polmatrix and sort according to the pol angles
        self.pol_matrix = rspHDU["SPECRESP POLMATRIX"].data
        self.pol_matrix = self.pol_matrix[:, sorted_indices, :]
        self.pol_matrix = self.pol_matrix.transpose()  # shape: (N_E, N_PA, N_SA)

        # read the unpol matrix
        self.unpol_matrix = rspHDU["SPECRESP UNPOLMATRIX"].data
        self.unpol_matrix = self.unpol_matrix.transpose()  # Shape: (N_E, N_SA)

        # pre interpolate the response for fitting
        # define params array (to be filled with tuples)
        self.fit_params = np.empty(self.unpol_matrix.shape, object)
        self._interpolate_rsp()

    def _interpolate_rsp(self):
        """
        Builds the interpolator for the response. This loops over all energies and each azimuthal bin
        to fit the 2nd degree harmonic function. The output is the best fit params which can be used to get
        response for any PA.

        """

        # fit the harmonics to the response and ready the interpolator
        # loop over energies
        for i in range(self.pol_matrix.shape[0]):
            # loop over scattering angles
            for j in range(self.pol_matrix.shape[2]):
                # get the counts as func of PA for this az bin
                counts = self.pol_matrix[i, :, j]

                # get the init params
                const = counts.mean()
                ampl = counts.max() - counts.min()
                phi = np.pi

                # fit
                popt, pcov = curve_fit(
                    harmonic, self.pol_ang, counts, p0=[const, ampl, phi, ampl, phi]
                )

                # fill the params array
                self.fit_params[i, j] = tuple(popt)

        # add one more dimention
        fit_pamas_3d = np.array(self.fit_params.tolist())

        # unpack along last axis
        self.const = fit_pamas_3d[..., 0]
        self.ampl1 = fit_pamas_3d[..., 1]
        self.phi1 = fit_pamas_3d[..., 2]
        self.ampl2 = fit_pamas_3d[..., 3]
        self.phi2 = fit_pamas_3d[..., 4]

    def evaluate_grid_point(self, pol_ang_val, pol_deg_val):
        """
        Evaluate the response at the given polarization angle and degree.

        pol_ang: The polarization angle to evaluate at.
        pol_deg: The polarization fraction to evaluate at.
        """

        # return the interpolated value at given pol_ang
        pol_matrix_val = harmonic(
            pol_ang_val, self.const, self.ampl1, self.phi1, self.ampl2, self.phi2
        )

        # compute at the given pol degree
        interp_rsp = (
            pol_deg_val / 100 * pol_matrix_val
            + (1 - pol_deg_val / 100) * self.unpol_matrix
        )

        return interp_rsp
