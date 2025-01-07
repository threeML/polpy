import numpy as np
from threeML.utils.OGIP.response import InstrumentResponse
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

class PolData(object):

    def __init__(self, polevents, polrsp, specrsp=None, reference_time=0.0):
        """
        container class that converts raw POLAR fits data into useful python
        variables

        This can build both the polarimetric and spectral data
        
        :param polevents: path to polarisation data event file
        :param specrsp: path to spectral response file
        :param polrsp: path to polarisation response file
                             it will use SABOUNDS to bin your SA data in 'polevents'
                             if 'NONE', we assume 'SA' data is already binned
        :param reference_time: reference time of the events (in SECOND)

        """
        
        # define global vars
        self.polevents = polevents
        self.specrsp = specrsp
        self.polrsp = polrsp
        
        if self.specrsp is not None:     
            with fits.open(self.specrsp) as hdu_spec:
                # This gets the spectral response
                mc_low = hdu_spec['MATRIX'].data.field('ENERG_LO')
                mc_high = hdu_spec['MATRIX'].data.field('ENERG_HI')
                ebounds = np.append(mc_low, mc_high[-1])
                matrix = hdu_spec['MATRIX'].data.field('MATRIX')
                matrix = matrix.transpose()

                # build the spectral response
                mc_energies = np.append(mc_low, mc_high[-1])
                self.rsp = InstrumentResponse(matrix=matrix, ebounds=ebounds, monte_carlo_energies=mc_energies)
    
        # open the event file
        hdu_evt = fits.open(self.polevents)
        
        # Extract mission and instrument info
        self.mission = hdu_evt['POLEVENTS'].header['TELESCOP']
        self.instrument = hdu_evt['POLEVENTS'].header['INSTRUME']
#        pha = hdu_evt['POLEVENTS'].data.field('ENERGY') old definition
        pha = hdu_evt['POLEVENTS'].data.field('CHANNEL')

        # non-zero ADC channels and correct energy range. Also bin the pha if using spectral response
        if 'ebounds' in locals():  # check if ebounds was defined
#            pha_mask2 = (pha <= ebounds.max()) & (pha >= ebounds.min())
#            pha_mask= pha_mask1 & pha_mask2
            # bin the ADC channels
#            self.pha = np.digitize(pha[pha_mask1 & pha_mask2], ebounds)
            self.n_channels= len(self.rsp.ebounds) - 1

        pha_mask = (pha >= 0)
        # get the dead time fraction
        self.dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[pha_mask]

        # get the arrival time, in SECOND
        self.time = (hdu_evt['POLEVENTS'].data.field('TIME'))[pha_mask] - reference_time

        # now do the scattering angles
        
        # there is some issue with applying the pha mask to this. Not consistent. To be checked !!
        scattering_angles = hdu_evt['POLEVENTS'].data.field('SA')

        # clear the bad scattering angles
        scat_angle_mask = scattering_angles != -1

        self.scattering_angle_time = (hdu_evt['POLEVENTS'].data.field('TIME'))[scat_angle_mask] - reference_time
        self.scattering_angle_dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[scat_angle_mask]
        self.scattering_angles = scattering_angles[scat_angle_mask]
        
        # read the instrument axis and source direction coordinates
        RA_X = hdu_evt['POLEVENTS'].header['RAX']*u.deg
        Dec_X = hdu_evt['POLEVENTS'].header['DECX']*u.deg
        RA_Z = hdu_evt['POLEVENTS'].header['RAZ']*u.deg
        Dec_Z = hdu_evt['POLEVENTS'].header['DECZ']*u.deg
        RA_S = hdu_evt['POLEVENTS'].header['RAGRB']*u.deg
        Dec_S = hdu_evt['POLEVENTS'].header['DECGRB']*u.deg
        
        # define the instrument direction vectors
        self._X = SkyCoord(RA_X, Dec_X, frame='icrs', obstime='J2000').cartesian
        self._Z = SkyCoord(RA_Z, Dec_Z, frame='icrs', obstime='J2000').cartesian
        
        # get the third axis
        self._Y = self._Z.cross(self._X)
        
        # define the source axis vector
        self._S = SkyCoord(RA_S, Dec_S, frame='icrs', obstime='J2000').cartesian

        # bin the scattering_angles
        if self.polrsp is not None:
            with fits.open(self.polrsp) as hdu_polrsp:
                example=hdu_polrsp['INEBOUNDS'].data.field('ENERG_LO')    
                samin = hdu_polrsp['SABOUNDS'].data.field('SA_MIN')
                samax = hdu_polrsp['SABOUNDS'].data.field('SA_MAX')
                scatter_bounds = np.append(samin, samax[-1])

                self.scattering_edges = scatter_bounds
                self.scattering_angles = np.digitize(self.scattering_angles, scatter_bounds)
                self.n_scattering_bins= len(self.scattering_edges) - 1

        else:
            self.scattering_edges = None
            self.scattering_angles = None
            
            


    def get_pa_offset(self) -> float:
        """ Compute the polarisation angle offset between local tangent frame and J2000 frame for
        *this* instrument.
        
        see docs for frame defn (add a link to docs)

        Returns:
            float: Polarisation angle offset between J2000 and local tangent frame.
        """

        # get the two transformation matrices
        R_IRF_J2000 = self._get_IRF_J2000_transform()
        R_LTP_XYZ = self._get_LTP_IRF_transform()
        
        R_LTP_J2000 = np.matmul(R_IRF_J2000, R_LTP_XYZ)
        
        # Compute the PA offset. This is basically azimuth of LTP Z-axis in J2000
        Z_LTP_J2000 = np.matmul(R_LTP_J2000, [0, 0, 1])
        psi = np.arctan2(Z_LTP_J2000[1], Z_LTP_J2000[0])

        # Return always between 0 to 360
        if psi < 0:
            psi += 2*np.pi

        return np.rad2deg(psi) % 180
        
    
    def _get_IRF_J2000_transform(self) -> np.ndarray:
        """ Returns the transformation matrix from instrument reference frame (IRF) to 
        J2000 frame.
        
        see docs for frame defn (add a link to docs)

        Returns:
            np.ndarray: Instrument to J2000 transformation matrix
        """

        # Matrix to go from XYZ to J2000 frame
        return np.array([self._X.get_xyz().value, self._Y.get_xyz().value, self._Z.get_xyz().value]).T


    def _get_LTP_IRF_transform(self) -> np.ndarray:
        """ Returns the transformation matrix from the local tangent plane (LTP) frame to 
        instrument reference frame (IRF).
        
        see docs for frame defn (add a link to docs)

        Returns:
            np.ndarray: LTP to IRF transformation matrix
        """
        
        # Compute source theta, phi
        # Compute the projection on XYZ
        ux = self._X.dot(self._S).value
        uy = self._Y.dot(self._S).value
        uz = self._Z.dot(self._S).value
        
        # Compute the theta,phi
        theta = np.arccos(uz)
        phi = np.arctan2(uy, ux)
        if phi < 0:
            phi += 2*np.pi

        # Matrix to go from NED to XYZ
        return np.array([[-np.cos(theta) * np.cos(phi), -np.sin(phi), -np.sin(theta) * np.cos(phi)],
                              [-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                              [np.sin(theta), 0, -np.cos(theta)]])
