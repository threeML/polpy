import collections
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import simpson

from astromodels import Parameter, Uniform_prior
from astromodels.core.model import Model
from polpy.polresponse import PolResponse
from threeML import PluginPrototype
from threeML.utils.binner import Rebinner


class PolarizationLike(PluginPrototype):
    """
    Preliminary PolPy polarization class
    """

    def __init__(self, name, observation, background, response, verbose=False):
        """

        The Polarization likelihood for PolPy. This plugin is heavily modeled off
        the 3ML dispersion based plugins. It interpolates the spectral photon model
        over the scattering angle bins to allow for spectral + polarization analysis.

        :param name: The name of the plugin
        :param observation: The POLAR observation file
        :param background: The POLAR background file
        :param response: The POLAR polarization response

        :param verbose:

        """

        # initialise required variables
        self._observation = observation
        self._background = background

        self._observed_counts = observation.counts.astype(np.int64)
        self._background_counts = background.counts
        self._background_count_errors = background.count_errors
        self._scale = observation.exposure / background.exposure
        self._exposure = observation.exposure
        self._background_exposure = background.exposure

        # for fitting (define vars, to be assigned later)
        self._likelihood_model = None
        self._pol_angle = None
        self._pol_degree = None

        # verbose check
        self._verbose = verbose

        # now do some double checks
        assert len(self._observed_counts) == len(self._background_counts)

        self._n_synthetic_datasets = 0

        # set up the effective area correction
        self._nuisance_parameter = Parameter(
            "cons_%s" % name,
            1.0,
            min_value=0.8,
            max_value=1.2,
            delta=0.05,
            free=False,
            desc="Effective area correction for %s" % name)

        nuisance_parameters = collections.OrderedDict()
        nuisance_parameters[self._nuisance_parameter.name] = self._nuisance_parameter

        # pass to the plugin proto
        super(PolarizationLike, self).__init__(name, nuisance_parameters)

        # The following vectors are the ones that will be really used for the computation. At the beginning they just
        # point to the original ones, but if a rebinner is used and/or a mask is created through set_active_measurements,
        # they will contain the rebinned and/or masked versions

        self._current_observed_counts = self._observed_counts
        self._current_background_counts = self._background_counts
        self._current_background_count_errors = self._background_count_errors

        # we can either attach or build a response
        assert isinstance(response, str) or isinstance(
            response, PolResponse), 'The response must be a file name or a PolarResponse'

        if isinstance(response, PolResponse):

            self._response = response

        else:

            self._response = PolResponse(response)

        # we also make sure the lengths match up here
        assert self._response.n_scattering_bins == len(
            self._observation.counts), 'observation counts shape does not agree with response shape'

    def use_effective_area_correction(self, lower=0.5, upper=1.5):
        """
        Use an area constant to correct for response issues

        :param lower:
        :param upper:
        :return:
        """

        self._nuisance_parameter.free = True
        self._nuisance_parameter.bounds = (lower, upper)
        self._nuisance_parameter.prior = Uniform_prior(
            lower_bound=lower, upper_bound=upper)
        if self._verbose:
            print('Using effective area correction')

    def fix_effective_area_correction(self, value=1):
        """

        fix the effective area correction to a particular values

        :param value:
        :return:
        """

        # allow the value to be outside the bounds
        if self._nuisance_parameter.max_value < value:

            self._nuisance_parameter.max_value = value + 0.1

        elif self._nuisance_parameter.min_value > value:

            self._nuisance_parameter.min_value = value = 0.1

        self._nuisance_parameter.fix = True
        self._nuisance_parameter.value = value

        if self._verbose:
            print('Fixing effective area correction')

    @property
    def effective_area_correction(self):

        return self._nuisance_parameter

    def set_model(self, likelihood_model_instance: Model):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        :param likelihood_model_instance: instance of Model
        :type likelihood_model_instance: astromodels.Model
        """

        if likelihood_model_instance is None:
            return

        for k, v in likelihood_model_instance.free_parameters.items():

            if 'polarization.degree' in k:
                self._pol_degree = v

            if 'polarization.angle' in k:
                self._pol_angle = v
        
        # assign  the model
        self._likelihood_model = likelihood_model_instance
    
    def _get_model_counts(self):

        # get the interpolated metrics for current pol_ang and pol_deg for each energy (this will be NE x NScat)
        interp_rsp = self._response.evaluate_grid_point(self._pol_angle.value, self._pol_degree.value)

        # get model flux and convole it with the interpolated rsp
        energies = self._response.ene_center
        model_flx = self._likelihood_model.get_point_source_fluxes(0, energies, tag=self._tag)

        # integrate
        model_rate = simpson(interp_rsp * model_flx[:, None], x=energies, axis=0)

        return self._nuisance_parameter.value * self._exposure * model_rate

    def get_log_like(self):

        model_counts = self._get_model_counts()

        model_counts += self._current_background_counts
        model_counts *= self._scale

        loglike = -(model_counts - self._current_observed_counts + self._current_observed_counts * np.log(self._current_observed_counts / model_counts))
        
        return np.nansum(loglike)


    def inner_fit(self):

        return self.get_log_like()
    
    @property
    def scattering_boundaries(self):
        """
        Energy boundaries of channels currently in use (rebinned, if a rebinner is active)

        :return: (sa_min, sa_max)
        """

        scattering_edges = np.array(self._observation.edges)

        sa_min, sa_max = scattering_edges[:-1], scattering_edges[1:]

        # if self._rebinner is not None:
        #     # Get the rebinned chans. NOTE: these are already masked

        #     sa_min, sa_max = self._rebinner.get_new_start_and_stop(
        #         sa_min, sa_max)

        return sa_min, sa_max

    @property
    def bin_widths(self):

        sa_min, sa_max = self.scattering_boundaries

        return sa_max - sa_min
    

    def display(self):
        """

        Display the data and model
        """
        
        # obs counts + err
        obs_cnt = self._observed_counts
        obs_cnt_err = np.sqrt(obs_cnt)
        
        # scaled bkg counts to source exposure + err
        scaled_bkg_cnt = self._background_counts * self._scale
        scaled_bkg_cnt_err = np.sqrt(self._background_counts) * self._scale
        
        # source counts + err (rate)
        source_cnt = obs_cnt - scaled_bkg_cnt
        source_cnt_err = np.sqrt(obs_cnt_err**2 + scaled_bkg_cnt_err**2)
        
        # get model counts
        model_cnt = self._get_model_counts()
        
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.errorbar(self.scattering_boundaries[0], source_cnt,
                     yerr=source_cnt_err, fmt='.', lw=1.5, capsize=3, label='Data', c='C4')
        ax.plot(self.scattering_boundaries[0], model_cnt, ds='steps-mid', lw=1.4, c='C3', label='Best-Fit Model')
        ax.legend(fontsize=12, loc='best')
        ax.set_xlabel('Scattering Angle (degrees)')
        ax.set_ylabel('Counts')
        ax.set_xlim(xmin=-10)
        fig.tight_layout()
        
        return fig


    @property
    def observation(self):
        return self._observation

    @property
    def background(self):
        return self._background

    @contextmanager
    def _without_rebinner(self):

        # Store rebinner for later use

        rebinner = self._rebinner

        # Clean mask and rebinning

        self.remove_rebinning()

        # Execute whathever

        yield

        # Restore mask and rebinner (if any)

        if rebinner is not None:

            # There was a rebinner, use it. Note that the rebinner applies the mask by itself

            self._apply_rebinner(rebinner)

    def rebin_on_background(self, min_number_of_counts):
        """
        Rebin the spectrum guaranteeing the provided minimum number of counts in each background bin. This is usually
        required for spectra with very few background counts to make the Poisson profile likelihood meaningful.
        Of course this is not relevant if you treat the background as ideal, nor if the background spectrum has
        Gaussian errors.

        The observed spectrum will be rebinned in the same fashion as the background spectrum.

        To neutralize this completely, use "remove_rebinning"

        :param min_number_of_counts: the minimum number of counts in each bin
        :return: none
        """

        # NOTE: the rebinner takes care of the mask already

        assert self._background is not None, "This data has no background, cannot rebin on background!"

        rebinner = Rebinner(self._background_counts,
                            min_number_of_counts, mask=None)

        self._apply_rebinner(rebinner)

    def rebin_on_source(self, min_number_of_counts):
        """
        Rebin the spectrum guaranteeing the provided minimum number of counts in each source bin.

        To neutralize this completely, use "remove_rebinning"

        :param min_number_of_counts: the minimum number of counts in each bin
        :return: none
        """

        # NOTE: the rebinner takes care of the mask already

        rebinner = Rebinner(self._observed_counts,
                            min_number_of_counts, mask=None)

        self._apply_rebinner(rebinner)

    def _apply_rebinner(self, rebinner):

        self._rebinner = rebinner

        # Apply the rebinning to everything.
        # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

        self._current_observed_counts, = self._rebinner.rebin(
            self._observed_counts)

        if self._background is not None:

            self._current_background_counts, = self._rebinner.rebin(
                self._background_counts)

            if self._background_count_errors is not None:
                # NOTE: the output of the .rebin method are the vectors with the mask *already applied*

                self._current_background_count_errors, = self._rebinner.rebin_errors(
                    self._background_count_errors)

        if self._verbose:
            print("Now using %s bins" % self._rebinner.n_bins)


    def remove_rebinning(self):
        """
        Remove the rebinning scheme set with rebin_on_background.

        :return:
        """

        self._rebinner = None

        self._current_observed_counts = self._observed_counts
        self._current_background_counts = self._background_counts
        self._current_background_count_errors = self._background_count_errors

