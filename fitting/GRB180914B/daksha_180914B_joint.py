#!/usr/bin/env python3  

import multiprocessing
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
trigger_time = 0
# fac3 0
daksha0_ts = TimeSeriesBuilder.from_polarization(name='daksha_pol0', polevents=f'/home/polpy/daksha_data/GRB180914B_PF_0.5_1.0x/GRB180914B_PA_068_PF_0.5_face_0_1.0x_dakshapol.pevt',
                                              polrsp=f'/home/polpy/daksha_data/GRB180914B_response/face_0/DAKSHA_POLRSP_EMIN_100_EMAX_1000_GRB180914B_0.prsp', specrsp=None,
                                               trigger_time=trigger_time)

# face 2
daksha2_ts = TimeSeriesBuilder.from_polarization(name='daksha_pol2', polevents=f'/home/polpy/daksha_data/GRB180914B_PF_0.5_1.0x/GRB180914B_PA_068_PF_0.5_face_2_1.0x_dakshapol.pevt',
                                              polrsp=f'/home/polpy/daksha_data/GRB180914B_response/face_2/DAKSHA_POLRSP_EMIN_100_EMAX_1000_GRB180914B_2.prsp', specrsp=None,
                                               trigger_time=trigger_time)

# face 3
daksha3_ts = TimeSeriesBuilder.from_polarization(name='daksha_pol3', polevents=f'/home/polpy/daksha_data/GRB180914B_PF_0.5_1.0x/GRB180914B_PA_068_PF_0.5_face_3_1.0x_dakshapol.pevt',
                                              polrsp=f'/home/polpy/daksha_data/GRB180914B_response/face_3/DAKSHA_POLRSP_EMIN_100_EMAX_1000_GRB180914B_3.prsp', specrsp=None,
                                                trigger_time=trigger_time)

# face 0
daksha0_ts.set_active_time_interval('-80 - 80')
daksha0_ts.set_background_interval('-245--90', '90-245')

# face 2
daksha2_ts.set_active_time_interval('-80 - 80')
daksha2_ts.set_background_interval('-245--90', '90-245')

# face 3
daksha3_ts.set_active_time_interval('-80 - 80')
daksha3_ts.set_background_interval('-245--90', '90-245')

daksha0_data = daksha0_ts.to_polarizationlike()
daksha2_data = daksha2_ts.to_polarizationlike()
daksha3_data = daksha3_ts.to_polarizationlike()


#effective area correction
#daksha0_data.use_effective_area_correction(0.1,300.0)
#daksha2_data.use_effective_area_correction(0.1,300.0)
#daksha3_data.use_effective_area_correction(0.1,300.0)

#setting up spectrum model
band = Band()

band.xp.prior = Uniform_prior(lower_bound=440, upper_bound=460)
band.xp.bounds = (None, None)
band.xp.value = 453
# band.xp.fixed = True

band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e1)
band.K.value = 0.056

band.alpha.bounds = (-2.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-0.75, sigma=0.05, lower_bound=-0.85, upper_bound=-0.65)
band.alpha.value = -0.75
# band.alpha.fixed = True

band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-2.10, sigma=0.05, lower_bound=-2.25, upper_bound=-1.95)
band.beta.value = -2.10
# band.beta.fixed = True

#settting up polarization model
lp = LinearPolarization(50,90)
lp.angle.prior = Uniform_prior(lower_bound=0.0, upper_bound=180.0)
lp.degree.prior = Uniform_prior(lower_bound=0.001, upper_bound=100.0)

#adding both component and defining the point source
sc =SpectralComponent('synch', band, lp)
ps = PointSource('GRB180914B',0,0, components = [sc])

combined_model = Model(ps)
datalist = DataList(daksha0_data, daksha2_data, daksha3_data)
#datalist = DataList(daksha3_data, daksha2_data)

combined_model.GRB180914B.spectrum.synch.Band.K.free = False
combined_model.GRB180914B.spectrum.synch.Band.alpha.free = False
combined_model.GRB180914B.spectrum.synch.Band.beta.free = False
combined_model.GRB180914B.spectrum.synch.Band.xp.free = False


# Setting up sampler and running bayes
bayes = BayesianAnalysis(combined_model,datalist)
bayes.set_sampler("multinest")
wrapped = [0] * len(combined_model.free_parameters)
bayes.sampler.setup(n_live_points=250,
                    resume = False,
                    importance_nested_sampling=False,
                    verbose=True,
                    wrapped_params=wrapped,
                    chain_name='chains/synch_p2')

# bayes.set_sampler("emcee")
# with multiprocessing.Pool(processes=20) as pool:
#     bayes.sampler.setup(nwalkers=1000000, n_iterations=100000, burnin=50, pool=pool,
#                     resume=False, verbose=True, chain_name='chains/synch_p2_emcee',
#                     wrapped_params=wrapped)
bayes.sample()


bayes.results.write_to(f"Daksha_polarization_results_GRB180914B_joint023_PF_0.5.fits", overwrite=True)
bayes.restore_median_fit()

#display everthing in bayes.results
bayes.results.display()

cornerplot = bayes.results.corner_plot()
cornerplot.savefig(f'GRB180914B_face_joint023_PF_0.5_corner_plot.png')
plt.close('all')

# display individual polarization data
modulationcurve = daksha0_data.display()
fig=modulationcurve.savefig(f'GRB180914B_face_face0_PF_0.5_modulation_curve.png')

modulationcurve = daksha2_data.display()
fig=modulationcurve.savefig(f'GRB180914B_face_face2_PF_0.5_modulation_curve.png')

modulationcurve = daksha3_data.display()
fig=modulationcurve.savefig(f'GRB180914B_face_face3_PF_0.5_modulation_curve.png')

print(f"Band XP ={combined_model.GRB180914B.spectrum.synch.Band.xp.value}",)
print(f"Band K ={combined_model.GRB180914B.spectrum.synch.Band.K.value}",)
print(f"Band Alpha ={combined_model.GRB180914B.spectrum.synch.Band.alpha.value}",)
print(f"Band Beta ={combined_model.GRB180914B.spectrum.synch.Band.beta.value}",)