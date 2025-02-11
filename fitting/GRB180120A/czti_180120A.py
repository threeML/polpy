
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import numpy as np
import os

np.seterr(all="ignore")

from threeML import *
silence_warnings()
set_threeML_style()


# reading polarization data from AstroSat CZTI and creating polarization plugin
trigger_time = 254120293.0

czti_polarization_ts = TimeSeriesBuilder.from_polarization(name='czti_pol', polevents='/home/polpy/polpy_test/polpy/fitting/GRB180120A/cztipol_GRB180120A.pevt',
                                              polrsp='/home/polpy/polpy_test/polpy/fitting/GRB180120A/CZTI_POLRSP_EMIN_110_EMAX_1000_GRB180120A.prsp', specrsp=None,
                                               trigger_time=trigger_time)

czti_polarization_ts.set_background_interval("-150.--50.","50.-150.")
czti_polarization_ts.set_active_time_interval('0-24.0')
czti_data = czti_polarization_ts.to_polarizationlike()


#effective area correction
#czti_data.fix_effective_area_correction(2.0)
czti_data.use_effective_area_correction(0.1,300.0)


gbm_detectors = ['n9','na','nb','b1']
source_interval='1.344-30.27'
background_interval='-50 --10,60-100'


dload = download_GBM_trigger_data("GRB180120207", detectors=gbm_detectors)


fluence_plugins = []
time_series = {}
#print(best_fit_model, "\n", source_interval, "\n", model.display())
for det in gbm_detectors:

    ts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(det, cspec_or_ctime_file=dload[det]["cspec"], rsp_file=dload[det]["rsp"])

    ts_cspec.set_background_interval(*background_interval.split(","))
    ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)

    ts_tte = TimeSeriesBuilder.from_gbm_tte(
        det,
        tte_file=dload[det]["tte"],
        rsp_file=dload[det]["rsp"],
        restore_background=f"{det}_bkg.h5",
    )

    time_series[det] = ts_tte

    ts_tte.set_active_time_interval(source_interval)

    ts_tte.view_lightcurve(-40, 160)

    fluence_plugin = ts_tte.to_spectrumlike()

    if det.startswith("b"):

        fluence_plugin.set_active_measurements("250-30000")

    else:

        fluence_plugin.set_active_measurements("9-900")

    fluence_plugin.rebin_on_background(1.0)

    fluence_plugins.append(fluence_plugin)


#setting up spectrum model
band = Band()

band.xp.prior = Log_normal(mu=np.log(147),sigma=np.log(100))
band.xp.bounds = (None, None)

band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E2)

band.alpha.bounds = (-1.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-0.29, sigma=0.5, lower_bound=-1.5, upper_bound=1.0)

band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-2.8,sigma=0.6, lower_bound=-7, upper_bound=-1.5)

#settting up polarization model
lp = LinearPolarization(60,47)
lp.angle.set_uninformative_prior(Uniform_prior)
lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)
lp.degree.value=60.
lp.angle.value=16.91

#adding both component and defining the point source
sc =SpectralComponent('synch', band, lp)
ps = PointSource('GRB180120A',0,0, components = [sc])

combined_model = Model(ps)
datalist = DataList(*fluence_plugins,czti_data)


# Setting up sampler and running bayes

bayes = BayesianAnalysis(combined_model,datalist)
bayes.set_sampler("multinest")
# wrapped = [0] * len(combined_model.free_parameters)
# wrapped[3] = 1
bayes.sampler.setup(n_live_points=400)#,
                        #    resume = False,
                        #    importance_nested_sampling=False,
                        #    verbose=True,
                        #    wrapped_params=wrapped,
                        #    chain_name='chains/synch_p2')
bayes.sample()


bayes.results.write_to("AstroSat_CZTI_polarization_results_GRB180120A.fits", overwrite=True)
bayes.restore_median_fit()


fig = display_spectrum_model_counts(bayes, step=False)


fig = display_spectrum_model_counts(bayes, min_rate=20)


#display everthing in bayes.results
bayes.results.display()


#plot_spectra(bayes.results)


cornerplot = bayes.results.corner_plot()


a = bayes.raw_samples
np.shape(a)
deg = a[:,5]
plt.hist(deg)
degrot = np.copy(deg)
degrot[deg > 97.64493707965887] -= 180
plt.hist(degrot)
print(np.mean(degrot), np.std(degrot))


modulationcurve = czti_data.display(show_model=True)
fig=modulationcurve.savefig('180120A_modulation_curve')



