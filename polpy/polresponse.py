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

    def __init__(self, response_file, pa_offset):
        """
        Construct the polarisation response from the mission specific polarisation response file.
:w
        :param response_file: Polarisation response file in the defined format (.prsp)
        :param pa_offset: Offset to be added to convert templates from LTP to J2000
        :returns: 
        :rtype: 

        """
        print(response_file)
        self._rsp_file = response_file
        self._pa_offset = pa_offset

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

            # we have 100% pol and 0% pol matrix in the prsp file
            pol_deg = np.array([0., 100.], dtype=np.float64)

            samin = np.array(hdu_pol['SABOUNDS'].data.field('SA_MIN'), dtype=np.float64)
            samax = np.array(hdu_pol['SABOUNDS'].data.field('SA_MAX'), dtype=np.float64)
            bins = np.append(samin, samax[-1])
            # get the bin centers as these are where things
            # should be evaluated
            bin_center = 0.5 * (bins[:-1] + bins[1:])

            polmatrix = hdu_pol['SPECRESP POLMATRIX'].data
            polmatrix = polmatrix.transpose()

            uppolmatrix = hdu_pol['SPECRESP UNPOLMATRIX'].data
            uppolmatrix = uppolmatrix.transpose()
            uppolmatrix = [uppolmatrix] * pol_ang.size
            uppolmatrix = np.stack(uppolmatrix, axis=1)

            pol_matrix = np.stack((uppolmatrix,polmatrix), axis=2)
            pol_matrix = np.array(pol_matrix, dtype=np.float64)

            all_interp = []

            # now we construct a series of interpolation
            # functions that are called during the fit.
            # we use some nice matrix math to handle this

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


    
