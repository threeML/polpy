#!/usr/bin/env python3  

import argparse
import warnings

warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import numpy as np

np.seterr(all="ignore")

from threeML import *
silence_warnings()
set_threeML_style()

grb = "GRB180103A"
pa = "122"
pf = 0.3
face =4  

# reading polarization data from AstroSat CZTI and creating polarization plugin
trigger_time = 0.0
daksha_ts = TimeSeriesBuilder.from_polarization(name=f'daksha_pol{face}', 
                                                polevents=f'/home/polpy/daksha_data/{grb}_PF_{pf}_1.0x/{grb}_PA_{pa}_PF_{pf}_face_{face}_1.0x_dakshapol.pevt',
                                                polrsp=f'/home/polpy/daksha_data/{grb}_response/face_{face}/DAKSHA_POLRSP_EMIN_100_EMAX_1000_{grb}_{face}.prsp', 
                                                specrsp=None,
                                                trigger_time=trigger_time)

daksha_ts.set_active_time_interval('-80 - 80')
daksha_ts.set_background_interval('-245--90', '90-245')
fig=daksha_ts.view_lightcurve(-250,250)
fig.savefig(f'{grb}_face_{face}_PF_{pf}_lightcurve.png')


daksha_data = daksha_ts.to_polarizationlike()

#setting up spectrum model
band = Band()

band.xp.prior = Truncated_gaussian(mu=273.00, sigma=20.0, lower_bound=200, upper_bound=350)
band.xp.bounds = (None, None)
band.xp.value = 273
# band.xp.fixed = True

band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e1)
band.K.value = 0.028

band.alpha.bounds = (-2.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-1.31, sigma=0.15, lower_bound=-2.0, upper_bound=-1.00)
band.alpha.value = -1.31
# band.alpha.fixed = True

band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-2.24, sigma=0.15, lower_bound=-3.0, upper_bound=-1.55)
band.beta.value = -2.24
# band.beta.fixed = True

#settting up polarization model
lp = LinearPolarization(50,90)
lp.angle.prior = Uniform_prior(lower_bound=0.0, upper_bound=180.0)
lp.degree.prior = Uniform_prior(lower_bound=0.001, upper_bound=100.0)

#adding both component and defining the point source
sc =SpectralComponent('synch', band, lp)
ps = PointSource('GRB180103A',0,0, components = [sc])

combined_model = Model(ps)
datalist = DataList(daksha_data)
#datalist = DataList(daksha3_data, daksha2_data)

# combined_model.GRB180103A.spectrum.synch.Band.K.free = False
# combined_model.GRB180103A.spectrum.synch.Band.alpha.free = False
# combined_model.GRB180103A.spectrum.synch.Band.beta.free = False
# combined_model.GRB180103A.spectrum.synch.Band.xp.free = False

# Setting up sampler and running bayes
bayes = BayesianAnalysis(combined_model, datalist)
bayes.set_sampler("multinest")
wrapped = [0] * len(combined_model.free_parameters)
bayes.sampler.setup(n_live_points=500,
                    resume=False,
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

bayes.results.write_to(f"Daksha_polarization_results_{grb}_face_{face}_PF_{pf}.fits", overwrite=True)

bayes.restore_median_fit()

#display everything in bayes.results
bayes.results.display()

cornerplot = bayes.results.corner_plot()
cornerplot.savefig(f'{grb}_face_{face}_PF_{pf}_corner_plot.png')
print("Corner plot saved successfully to:", f'{grb}_face_{face}_PF_{pf}_corner_plot.png')
plt.close('all')

# display individual polarization data
modulationcurve = daksha_data.display()
modulationcurve.savefig(f'{grb}_face_{face}_PF_{pf}_modulation_curve.png')