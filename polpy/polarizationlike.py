import collections
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import numba as nb

from astromodels import Parameter, Uniform_prior
from polpy.polresponse import PolResponse
from threeML import PluginPrototype
from threeML.io.plotting.step_plot import step_plot
from threeML.utils.binner import Rebinner
from threeML.utils.polarization.binned_polarization import \
    BinnedModulationCurve
from threeML.utils.statistics.likelihood_functions import (
    poisson_observed_gaussian_background, poisson_observed_poisson_background)


class PolarizationLike(PluginPrototype):
    """
    Preliminary POLAR polarization plugin
    """

    def __init__(self, name, observation, background, response, interval_number=None, verbose=False):
        """

        The Polarization likelihood for POLAR. This plugin is heavily modeled off
        the 3ML dispersion based plugins. It interpolates the spectral photon model
        over the scattering angle bins to allow for spectral + polarization analysis.



        :param interval_number: The time interval starting from 1.
        :param name: The name of the plugin
        :param observation: The POLAR observation file
        :param background: The POLAR background file
        :param response: The POLAR polarization response

        :param verbose:

        """
        # attach the required variables

        self._observation = observation
        self._background = background

        self._observed_counts = observation.counts.astype(np.int64)
        self._background_counts = background.counts
        self._background_count_errors = background.count_errors
        self._scale = observation.exposure / background.exposure
        self._exposure = observation.exposure
        self._background_exposure = background.exposure

        self._likelihood_model = None
        self._rebinner = None

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

        self._verbose = verbose

        # we can either attach or build a response

        assert isinstance(response, str) or isinstance(
            response, PolResponse), 'The response must be a file name or a PolarResponse'

        if isinstance(response, PolResponse):

            self._response = response

        else:

            self._response = PolResponse(response)

        # attach the interpolators to the

        self._all_interp = self._response.interpolators

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

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        :param likelihood_model_instance: instance of Model
        :type likelihood_model_instance: astromodels.Model
        """

        if likelihood_model_instance is None:
            return

        # if self._source_name is not None:

        #     # Make sure that the source is in the model
        #     assert self._source_name in likelihood_model_instance.sources, \
        #                                         "This XYLike plugin refers to the source %s, " \
        #                                         "but that source is not in the likelihood model" % (self._source_name)

        for k, v in likelihood_model_instance.free_parameters.items():

            if 'polarization.degree' in k:
                self._pol_degree = v

            if 'polarization.angle' in k:
                self._pol_angle = v

        # now we need to get the integral flux

        _, integral = self._get_diff_flux_and_integral(
            likelihood_model_instance)

        self._integral_flux = integral

        self._likelihood_model = likelihood_model_instance

    def _get_diff_flux_and_integral(self, likelihood_model):

        n_point_sources = likelihood_model.get_number_of_point_sources()

        # Make a function which will stack all point sources (OGIP do not support spatial dimension)

        def differential_flux(scattering_edges):
            fluxes = likelihood_model.get_point_source_fluxes(
                0, scattering_edges, tag=self._tag)

            # If we have only one point source, this will never be executed
            for i in range(1, n_point_sources):
                fluxes += likelihood_model.get_point_source_fluxes(
                    i, scattering_edges, tag=self._tag)

            return fluxes

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # scattering_edges. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        def integral(e1, e2):
            # Simpson's rule

            return (e2 - e1) / 6.0 * (differential_flux(e1) + 4 * differential_flux(
                (e1 + e2) / 2.0) + differential_flux(e2))

        return differential_flux, integral

    def _get_model_rate(self):

        # first we need to get the integrated expectation from the spectrum

        intergal_spectrum = np.array(
            [self._integral_flux(emin, emax) for emin, emax in zip(self._response.ene_lo, self._response.ene_hi)])

        # we evaluate at the center of the bin. the bin widths are already included
        eval_points = np.array(
            [[ene, self._pol_angle.value, self._pol_degree.value] for ene in self._response.energy_mid])

        # expectation = []
        

        # # create the model counts by summing over energy

        # for i, interpolator in enumerate(self._all_interp):
        #     rate = np.dot(interpolator(eval_points), intergal_spectrum)

        #     expectation.append(rate)


        return _interpolate_all(self._all_interp, intergal_spectrum, eval_points)

    def _get_model_counts(self):

        if self._rebinner is None:
            model_rate = self._get_model_rate()

        else:

            model_rate, = self._rebinner.rebin(self._get_model_rate())

        return self._nuisance_parameter.value * self._exposure * model_rate

    def get_log_like(self):

        model_counts = self._get_model_counts()

        if self._background.is_poisson:

            loglike, bkg_model = poisson_observed_poisson_background(
                self._current_observed_counts, self._current_background_counts, self._scale, model_counts)

        else:

            loglike, bkg_model = poisson_observed_gaussian_background(
                self._current_observed_counts, self._current_background_counts, self._current_background_count_errors,
                model_counts)

        return np.sum(loglike)

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

        if self._rebinner is not None:
            # Get the rebinned chans. NOTE: these are already masked

            sa_min, sa_max = self._rebinner.get_new_start_and_stop(
                sa_min, sa_max)

        return sa_min, sa_max

    @property
    def bin_widths(self):

        sa_min, sa_max = self.scattering_boundaries

        return sa_max - sa_min

    def display(self,
                ax=None,
                show_data=True,
                show_model=True,
                show_total=False,
                model_kwargs={},
                data_kwargs={},
                edges=True,
                min_rate=None):
        """

        Display the data, model, or both.

        :param ax:
        :param show_data:
        :param show_model:
        :param show_total:
        :param model_kwargs:
        :param data_kwargs:
        :return:
        """

        tmp = ((self._observed_counts / self._exposure) -
               self._background_counts / self._background_exposure)

        scattering_edges = np.array(self._observation.edges)

        sa_min, sa_max = scattering_edges[:-1], scattering_edges[1:]

        tmp_db = ((self._observed_counts / self._exposure) - self._background_counts / self._background_exposure) / (
            sa_max - sa_min)

        old_rebinner = self._rebinner

        if min_rate is not None:

            rebinner = Rebinner(tmp_db, min_rate, mask=None)

            self._apply_rebinner(rebinner)

            net_rate = rebinner.rebin(tmp)
        else:

            net_rate = tmp

        sa_min, sa_max = self.scattering_boundaries

        if show_total:
            show_model = False
            show_data = False

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        xs = self.scattering_boundaries

        if show_total:

            total_rate = self._current_observed_counts / self._exposure / self.bin_widths

            bkg_rate = self._current_background_counts / \
                self._background_exposure / self.bin_widths

            total_errors = np.sqrt(total_rate)

            if self._background.is_poisson:

                bkg_errors = np.sqrt(bkg_rate)

            else:

                bkg_errors = self._current_background_count_errors / self.bin_widths

            ax.hlines(total_rate, sa_min, sa_max,
                      color='#7D0505', **data_kwargs)
            ax.vlines(
                np.mean([xs], axis=1),
                total_rate - total_errors,
                total_rate + total_errors,
                color='#7D0505',
                **data_kwargs)

            ax.hlines(bkg_rate, sa_min, sa_max, color='#0D5BAE', **data_kwargs)
            ax.vlines(
                np.mean([xs], axis=1), bkg_rate - bkg_errors, bkg_rate + bkg_errors, color='#0D5BAE', **data_kwargs)

        if show_data:

            if self._background.is_poisson:

                errors = np.sqrt((self._current_observed_counts / self._exposure**2 / self.bin_widths**2) +
                                 (self._current_background_counts / self._background_exposure**2/ self.bin_widths**2))

            else:

                errors = np.sqrt((self._current_observed_counts / self._exposure**2 / self.bin_widths**2) +
                                 (self._current_background_count_errors / self._background_exposure/ self.bin_widths)**2)

            ax.hlines(net_rate / self.bin_widths,
                      sa_min, sa_max, **data_kwargs)
            ax.vlines(
                np.mean([xs], axis=1), (net_rate - errors) /
                self.bin_widths, (net_rate + errors) / self.bin_widths,
                **data_kwargs)

        if show_model:

            if edges:

                step_plot(
                    ax=ax,
                    xbins=np.vstack([sa_min, sa_max]).T,
                    y=self._get_model_counts() / self._exposure / self.bin_widths,
                    **model_kwargs)

            else:

                y = self._get_model_counts() / self._exposure / self.bin_widths
                ax.hlines(y, sa_min, sa_max, **model_kwargs)

        ax.set_xlabel('Scattering Angle')
        ax.set_ylabel('Net Rate (cnt/s/bin)')

        if old_rebinner is not None:

            # There was a rebinner, use it. Note that the rebinner applies the mask by itself

            self._apply_rebinner(old_rebinner)

        else:

            self.remove_rebinning()

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




@nb.njit(fastmath=True)
def _interpolate_all(interpolators, integral_spectrum, eval_points):

    N = len(interpolators)
    expectation = np.empty(N)

    
    
    for n in range(N):
    
        expectation[n] = np.dot(interpolators[n].evaluate(eval_points), integral_spectrum )

    return expectation
