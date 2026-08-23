import numpy as np
from threeML.utils.OGIP.response import InstrumentResponse
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


class PolData(object):

    def __init__(self, polevents, reference_time=0.0):
        """
        container class that converts raw polarisation fits data into useful python
        variables

        This can build both the polarimetric and spectral data

        :param polevents: path to polarisation data event file
        :param reference_time: reference time of the events (in SECOND)

        """

        # define global vars
        self.polevents = polevents

        # open the event file
        hdu_evt = fits.open(self.polevents)

        # Extract header info
        self.mission = hdu_evt["POLEVENTS"].header["TELESCOP"]
        self.instrument = hdu_evt["POLEVENTS"].header["INSTRUME"]
        self.n_channels = hdu_evt["POLEVENTS"].header["NCHANS"]
        self.n_scattering_bins = hdu_evt["POLEVENTS"].header["NSABINS"]

        # get scattering binsize and define bin edges
        scat_bin_size = 360 / self.n_scattering_bins
        self.scattering_edges = np.arange(0, 360.1, scat_bin_size)

        # extract pha and scattering angles
        pha = hdu_evt["POLEVENTS"].data.field("CHANNEL")
        scattering_angles = hdu_evt["POLEVENTS"].data.field("SABIN")
        print("scattering angles :", scattering_angles)

        # get the dead time fraction
        dead_time_fraction = hdu_evt["POLEVENTS"].data.field("DEADFRAC")

        # get the arrival time, in SECOND
        time = (hdu_evt["POLEVENTS"].data.field("TIME")) - reference_time

        # mask bad pha and scattering angles
        pha_mask = pha >= 0
        scat_angle_mask = scattering_angles != -1
        total_mask = pha_mask & scat_angle_mask  # this should be looked into later

        # clear the bad scattering angles angles and pha
        self.pha = pha[pha_mask]
        self.time = time[pha_mask]
        self.dead_time_fraction = dead_time_fraction[pha_mask]
        self.scattering_angles = scattering_angles[total_mask]
        self.scattering_angle_time = time[total_mask]
        self.scattering_angle_dead_time_fraction = dead_time_fraction[total_mask]

        # read the instrument axis and source direction coordinates
        RA_X = hdu_evt["POLEVENTS"].header["RAX"] * u.deg
        Dec_X = hdu_evt["POLEVENTS"].header["DECX"] * u.deg
        RA_Z = hdu_evt["POLEVENTS"].header["RAZ"] * u.deg
        Dec_Z = hdu_evt["POLEVENTS"].header["DECZ"] * u.deg
        RA_S = hdu_evt["POLEVENTS"].header["RAGRB"] * u.deg
        Dec_S = hdu_evt["POLEVENTS"].header["DECGRB"] * u.deg

        # define the instrument direction vectors
        self._X = SkyCoord(RA_X, Dec_X, frame="icrs", obstime="J2000").cartesian
        self._Z = SkyCoord(RA_Z, Dec_Z, frame="icrs", obstime="J2000").cartesian

        # get the third axis
        self._Y = self._Z.cross(self._X)

        # define the source axis vector
        self._S = SkyCoord(RA_S, Dec_S, frame="icrs", obstime="J2000")

    def get_pa_offset(self) -> float:
        """Compute the polarisation angle offset between local tangent frame and the X (North) axis of the IAU
        frame for *this* instrument.

        see docs for frame defn (add a link to docs)

        Returns:
            float: Polarisation angle offset between J2000 and local tangent frame.
        """

        # Get the direction of local north at source position (IAU "X" axis)
        # define local north
        if self._S.dec >= 0:
            N_IAU = SkyCoord(
                180.0 * u.deg + self._S.ra,
                90 * u.deg - self._S.dec,
                frame="icrs",
                obstime="J2000",
            ).cartesian.xyz.value
        else:
            N_IAU = SkyCoord(
                self._S.ra, 90 * u.deg + self._S.dec, frame="icrs", obstime="J2000"
            ).cartesian.xyz.value

        # get the two transformation matrices
        R_J2000_IRF = self._get_J2000_IRF_transform()
        R_IRF_LTP = self._get_IRF_LTP_transform()

        R_J2000_LTP = np.matmul(R_IRF_LTP, R_J2000_IRF)

        # Compute the PA offset. This is basically azimuth of LTP Z-axis in J2000
        Z_LTP_J2000 = np.matmul(R_J2000_LTP, N_IAU)
        psi_0 = np.arctan2(Z_LTP_J2000[1], Z_LTP_J2000[0])

        return np.rad2deg(psi_0)

    def _get_J2000_IRF_transform(self) -> np.ndarray:
        """Returns the transformation matrix from the J2000 frame to
        the instrument reference frame (IRF) frame.

        see docs for frame defn (add a link to docs)

        Returns:
            np.ndarray: Instrument to J2000 transformation matrix
        """

        # Matrix to go from J2000 to IRF
        return np.array(
            [self._X.get_xyz().value, self._Y.get_xyz().value, self._Z.get_xyz().value]
        )

    def _get_IRF_LTP_transform(self) -> np.ndarray:
        """Returns the transformation matrix from the instrument reference frame (IRF) to the
        local tangent plane (LTP) frame.

        see docs for frame defn (add a link to docs)

        Returns:
            np.ndarray: LTP to IRF transformation matrix
        """

        # Compute source theta, phi
        # Compute the projection on XYZ
        ux = self._X.dot(self._S.cartesian).value
        uy = self._Y.dot(self._S.cartesian).value
        uz = self._Z.dot(self._S.cartesian).value

        # Compute the theta,phi
        theta = np.arccos(uz)
        phi = np.arctan2(uy, ux)
        if phi < 0:
            phi += 2 * np.pi

        print("theta, phi:", np.rad2deg(theta), np.rad2deg(phi))

        # Matrix to go from IRF to LTP (NED) frame
        # get the local north
        if theta < np.pi / 2:
            north = np.array(
                [
                    np.sin(np.pi / 2 - theta) * np.cos(phi + np.pi),
                    np.sin(np.pi / 2 - theta) * np.sin(phi + np.pi),
                    np.cos(np.pi / 2 - theta),
                ]
            )
        else:
            north = np.array(
                [
                    np.sin(theta - np.pi / 2) * np.cos(phi),
                    np.sin(theta - np.pi / 2) * np.sin(phi),
                    np.cos(theta - np.pi / 2),
                ]
            )

        # get the local east
        east = np.array([-np.sin(phi), np.cos(phi), 0])

        # get the source direction (for Down component)
        source = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        return np.array([north, east, -1 * source])
