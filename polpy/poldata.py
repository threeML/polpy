import numpy as np
from threeML.utils.OGIP.response import InstrumentResponse
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

class PolData(object):

    def __init__(self, polevents, polrsp, specrsp=None, reference_time=0.0):
        """
        container class that converts raw polarisation fits data into useful python
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


        
        if self.polrsp is not None:
            with fits.open(self.polrsp) as hdu_pol:
                mc_low=hdu_pol['INEBOUNDS'].data.field('ENERG_LO')
                mc_high=hdu_pol['INEBOUNDS'].data.field('ENERG_HI')
                ebounds = np.append(mc_low, mc_high[-1])
                self.n_channels = hdu_pol['INEBOUNDS'].header['NAXIS2']
                # self.n_channels = hdu_pol['INEBOUNDS'].header['EBINS'] to be used when response is properly generated
        elif self.specrsp is not None:    
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
        if 'ENERGY' in hdu_evt['POLEVENTS'].data.names:
            pha = hdu_evt['POLEVENTS'].data.field('ENERGY')
            # non-zero ADC channels and correct energy range. Also bin the pha if using spectral response
            if 'ebounds' in locals():  # check if ebounds was defined
                pha_mask1 = (pha > 0)
                pha_mask2 = (pha <= ebounds.max()) & (pha >= ebounds.min())
                pha_mask = pha_mask1 & pha_mask2
                # bin the ADC channels is not needed anymore since we have channels already.
                self.pha = np.digitize(pha[pha_mask], ebounds)
            else:
                pha_mask = (pha >= 0)
                self.pha = pha[pha_mask]
            print("PHA:",self.pha)
        else:
            pha = hdu_evt['POLEVENTS'].data.field('CHANNEL')
            if 'ebounds' in locals():  # check if ebounds was defined
                # bin the ADC channels is not needed anymore since we have channels already.
                self.n_channels = len(ebounds) - 1
                pha_mask = (pha >= 0)
                self.pha = pha[pha_mask]
            print("PHA:",self.pha)
        print("length of pha:",self.n_channels)
        # get the dead time fraction
        self.dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[pha_mask]

        # get the arrival time, in SECOND
        self.time = (hdu_evt['POLEVENTS'].data.field('TIME'))[pha_mask] - reference_time
        
        # now do the scattering angles
        
        # there is some issue with applying the pha mask to this. Not consistent. To be checked !!
        if 'SA' in hdu_evt['POLEVENTS'].data.names:
            scattering_angles = hdu_evt['POLEVENTS'].data.field('SA')
            print("scattering angles :",scattering_angles)
        else:
            scattering_angles = hdu_evt['POLEVENTS'].data.field('SABIN')
            print("scattering angles :",scattering_angles)

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
        self._S = SkyCoord(RA_S, Dec_S, frame='icrs', obstime='J2000')

        # bin the scattering_angles
        if self.polrsp is not None:
            with fits.open(self.polrsp) as hdu_pol:  
                samin = hdu_pol['SABOUNDS'].data.field('SA_MIN')
                samax = hdu_pol['SABOUNDS'].data.field('SA_MAX')
                scatter_bounds = np.append(samin, samax[-1])

                self.scattering_edges = scatter_bounds
                self.scattering_angles = np.digitize(self.scattering_angles, scatter_bounds)
                self.n_scattering_bins= len(self.scattering_edges) - 1
                #self.n_scattering_bins= hdu_pol['INPAVALS'].header['PABINS'] to be used when response is properly generated

        else:
            self.scattering_edges = None
            self.scattering_angles = None
            
            


    def get_pa_offset(self) -> float:
        """ Compute the polarisation angle offset between local tangent frame and the X (North) axis of the IAU 
        frame for *this* instrument.
        
        see docs for frame defn (add a link to docs)

        Returns:
            float: Polarisation angle offset between J2000 and local tangent frame.
        """

        # Get the direction of local north at source position (IAU "X" axis)
        N_IAU = [-np.sin(self._S.dec.rad)*np.cos(self._S.ra.rad),
                 -np.sin(self._S.dec.rad)*np.sin(self._S.ra.rad),
                 np.cos(self._S.dec.rad)]
        
        # get the two transformation matrices
        R_J2000_IRF = self._get_J2000_IRF_transform()
        R_IRF_LTP = self._get_IRF_LTP_transform()
        
        R_J2000_LTP = np.matmul(R_IRF_LTP, R_J2000_IRF)
        
        # Compute the PA offset. This is basically azimuth of LTP Z-axis in J2000
        Z_LTP_J2000 = np.matmul(R_J2000_LTP, N_IAU)
        psi_0 = np.arctan2(Z_LTP_J2000[1], Z_LTP_J2000[0])
        
        print("PA offset:", np.rad2deg(psi_0))

        return np.rad2deg(psi_0)
        
    
    def _get_J2000_IRF_transform(self) -> np.ndarray:
        """ Returns the transformation matrix from the J2000 frame to 
        the instrument reference frame (IRF) frame.
        
        see docs for frame defn (add a link to docs)

        Returns:
            np.ndarray: Instrument to J2000 transformation matrix
        """

        # Matrix to go from J2000 to IRF
        return np.array([self._X.get_xyz().value, self._Y.get_xyz().value, self._Z.get_xyz().value])


    def _get_IRF_LTP_transform(self) -> np.ndarray:
        """ Returns the transformation matrix from the instrument reference frame (IRF) to the 
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
            phi += 2*np.pi
            
        print("theta, phi:", np.rad2deg(theta), np.rad2deg(phi))

        # Matrix to go from IRF to LTP
        return np.array([[-np.cos(theta) * np.cos(phi), -np.sin(phi), -np.sin(theta) * np.cos(phi)],
                              [-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                              [np.sin(theta), 0, -np.cos(theta)]]).T
