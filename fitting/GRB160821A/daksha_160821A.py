
from threeML import *
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import numpy as np
import os

np.seterr(all="ignore")

silence_warnings()
set_threeML_style()


# reading polarization data from AstroSat CZTI and creating polarization plugin
trigger_time = 0

czti_polarization_ts = TimeSeriesBuilder.from_polarization(name='daksha_pol', polevents='/home/polpy/polpy_test/polpy/data/GRB160821A_PA_025_PF_0.5_face_0_1.0x_dakshapol.pevt',
                                              polrsp='/home/polpy/polpy_test/polpy/data/DAKSHA_POLRSP_EMIN_100_EMAX_1000_GRB160821A_0.prsp', specrsp=None,
                                               trigger_time=trigger_time)


czti_polarization_ts.set_active_time_interval('-21 - 21 ')
czti_polarization_ts.set_background_interval('-250--30')

czti_data = czti_polarization_ts.to_polarizationlike()


#effective area correction
#czti_data.fix_effective_area_correction(2.0)
czti_data.use_effective_area_correction(0.1,300.0)



#setting up spectrum model
band = Band()

band.xp.prior = Uniform_prior(lower_bound=935, upper_bound=945)
band.xp.bounds = (None, None)

band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E2)

band.alpha.bounds = (-1.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-1.05, sigma=0.05, lower_bound=-1.10, upper_bound=-1.00)

band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-2.30, sigma=0.05, lower_bound=-2.35, upper_bound=-2.25)

#settting up polarization model
lp = LinearPolarization(0,0)
lp.angle.set_uninformative_prior(Uniform_prior)
lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)

#adding both component and defining the point source
sc =SpectralComponent('synch', band, lp)
ps = PointSource('GRB160821A',0,0, components = [sc])

combined_model = Model(ps)
datalist = DataList(czti_data)


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


bayes.results.write_to("Daksha_polarization_results_GRB160821A_face_0_PF_0.5.fits", overwrite=True)
bayes.restore_median_fit()


fig = display_spectrum_model_counts(bayes, step=False)


fig = display_spectrum_model_counts(bayes, min_rate=20)


#display everthing in bayes.results
bayes.results.display()


#plot_spectra(bayes.results)


cornerplot = bayes.results.corner_plot()
cornerplot.savefig('160821A_face_0_PF_0.5_corner_plot.png')

#a = bayes.raw_samples
#np.shape(a)
#deg = a[:,5]
#plt.hist(deg)
#degrot = np.copy(deg)
#degrot[deg > 97.64493707965887] -= 180
#plt.hist(degrot)
#print(np.mean(degrot), np.std(degrot))
#

modulationcurve = czti_data.display(show_model=True)
fig=modulationcurve.savefig('160821A_face_0_PF_0.5_modulation_curve.png')



