#!/usr/bin/env python3  

import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import numpy as np

np.seterr(all="ignore")

from threeML import *
silence_warnings()
set_threeML_style()


# Input parameters
GRB_NAME = "GRB171010A"
PA = "040"
PF = "0.8"
FACES = [12,5,1]
DATA_DIR = "/home/polpy/daksha_data"
TRIGGER_TIME = 0

# Initialize data list
datalist_items = []
trigger_time = TRIGGER_TIME

# Create time series for each face
for face in FACES:
  ts = TimeSeriesBuilder.from_polarization(
    name=f'daksha_pol{face}',
    polevents=f'{DATA_DIR}/{GRB_NAME}_PF_{PF}_1.0x/{GRB_NAME}_PA_{PA}_PF_{PF}_face_{face}_1.0x_dakshapol.pevt',
    polrsp=f'{DATA_DIR}/{GRB_NAME}_response/face_{face}/DAKSHA_POLRSP_EMIN_100_EMAX_1000_{GRB_NAME}_{face}.prsp',
    specrsp=None,
    trigger_time=trigger_time
  )
  
  ts.set_active_time_interval('-50 - 50')
  ts.set_background_interval('-245--90', '90-245')
  datalist_items.append(ts.to_polarizationlike())

#setting up spectrum model
band = Band()

band.xp.prior = Truncated_gaussian(mu=228.77, sigma=10.0, lower_bound=200, upper_bound=300)
band.xp.bounds = (None, None)
band.xp.value = 273
# band.xp.fixed = True

band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e1)
band.K.value = 0.11693

band.alpha.bounds = (-2.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-0.91, sigma=0.15, lower_bound=-1.5, upper_bound=-0.50)
band.alpha.value = -1.31
# band.alpha.fixed = True

band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-2.28, sigma=0.15, lower_bound=-2.50, upper_bound=-2.00)
band.beta.value = -2.24
# band.beta.fixed = True

#settting up polarization model
lp = LinearPolarization(50,90)
lp.angle.prior = Uniform_prior(lower_bound=0.0, upper_bound=180.0)
lp.degree.prior = Uniform_prior(lower_bound=0.001, upper_bound=100.0)

#adding both component and defining the point source
sc =SpectralComponent('synch', band, lp)
ps = PointSource('GRB171010A',0,0, components = [sc])

combined_model = Model(ps)
datalist = DataList(*datalist_items)

# Freeze parameters
combined_model[GRB_NAME].spectrum.synch.Band.K.free = True
combined_model[GRB_NAME].spectrum.synch.Band.alpha.free = True
combined_model[GRB_NAME].spectrum.synch.Band.beta.free = True
combined_model[GRB_NAME].spectrum.synch.Band.xp.free = True

# Bayesian analysis
bayes = BayesianAnalysis(combined_model, datalist)
bayes.set_sampler("multinest")
wrapped = [0] * len(combined_model.free_parameters)
bayes.sampler.setup(n_live_points=1000, resume=False, importance_nested_sampling=False,
          verbose=True, wrapped_params=wrapped, chain_name='chains/synch_p2')
bayes.sample()

# Save and display results
faces_str = ''.join(map(str, FACES))
# bayes.results.write_to(f"Daksha_polarization_results_{GRB_NAME}_joint{faces_str}_PF_{PF}.fits", overwrite=True)
bayes.restore_median_fit()
bayes.results.display()

cornerplot = bayes.results.corner_plot()
cornerplot.savefig(f'{GRB_NAME}_face_joint{faces_str}_PF_{PF}_corner_plot.png')
print("corner plot saved at:", f'{GRB_NAME}_face_joint{faces_str}_PF_{PF}_corner_plot.png')
plt.close('all')

for face in FACES:
  modulationcurve = datalist_items[FACES.index(face)].display()
  modulationcurve.savefig(f'{GRB_NAME}_face{face}_PF_{PF}_modulation_curve.png')

print(f"Band XP = {combined_model[GRB_NAME].spectrum.synch.Band.xp.value}")
print(f"Band K = {combined_model[GRB_NAME].spectrum.synch.Band.K.value}")
print(f"Band Alpha = {combined_model[GRB_NAME].spectrum.synch.Band.alpha.value}")
print(f"Band Beta = {combined_model[GRB_NAME].spectrum.synch.Band.beta.value}")
