#!/usr/bin/env python

"""
An example script which shows how to to jointly fit polarization data sets from two different
instruments with POLAR and Daksha simulated data.

The injected PA and PF values used for this example are 50 % and 117 degrees respectively.

"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# imports for modelling
from astromodels.core.model import Model, PointSource
from astromodels.core.polarization import LinearPolarization
from astromodels.core.spectral_component import SpectralComponent
from astromodels.functions.functions_1D.powerlaws import Band
from astromodels.functions.priors import (
    Uniform_prior,
    Truncated_gaussian,
    Log_uniform_prior,
)

# threeML import of necessary class
from threeML import silence_warnings, set_threeML_style
from threeML.utils.data_builders import TimeSeriesBuilder
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis

warnings.simplefilter("ignore")
silence_warnings()
set_threeML_style()

# get data path
datapath = Path(__file__).parent.parent.joinpath("tests/data")

# Load POLAR data and response
polar_trigger_time = 1474786761.0

# polarization
polar_polarization_ts = TimeSeriesBuilder.from_polarization(
    "polar_pol",
    polevents=datapath.joinpath(
        "GRB160325A_PA131_PF0.5/POLAR_160325A-N010-P050-A117.pevt"
    ).as_posix(),
    polrsp=datapath.joinpath("GRB160325A_PA131_PF0.5/POLAR_160325A.prsp").as_posix(),
    trigger_time=polar_trigger_time,
)

# view POLAR LC
polar_polarization_ts.time_series.view_lightcurve(-300, 300)
plt.title("POLAR")
plt.show()

# set background interval
polar_polarization_ts.set_background_interval("-250.0--20.0", "70-250")

# set source interval
polar_polarization_ts.set_active_time_interval("0.5-40.0")

# view POLAR LC
polar_polarization_ts.time_series.view_lightcurve(-300, 300)
plt.title("POLAR")
plt.show()

# convert to polarization object
polar_data = polar_polarization_ts.to_polarizationlike()


# Load Daksha data and response
datalist_daksha = []
trigger_time = 0

FACES = [11, 4, 12, 0, 1]

# Create time series for each face
for face in FACES:
    ts = TimeSeriesBuilder.from_polarization(
        name=f"daksha_pol{face}",
        polevents=datapath.joinpath(
            f"GRB160325A_PA131_PF0.5/GRB160325A_PA_131_PF_0.5_face_{face:d}_10.0x_dakshapol.pevt"
        ).as_posix(),
        polrsp=datapath.joinpath(
            f"GRB160325A_PA131_PF0.5/DAKSHA_POLRSP_EMIN_100_EMAX_1000_GRB160325A_{face:0}.prsp"
        ).as_posix(),
        trigger_time=trigger_time,
    )

    # view LC for this face
    ts.time_series.view_lightcurve(-250, 250)
    plt.title(f"Daksha Face: {face:d}")
    plt.show()

    # set background interval
    ts.set_background_interval("-200.--50.", "50.-200.")

    # set source interval
    ts.set_active_time_interval("-20. - 20.")

    # view LC after selection
    ts.time_series.view_lightcurve(-250, 250)
    plt.title(f"Daksha Face: {face:d}")
    plt.show()

    # append to datalist
    datalist_daksha.append(ts.to_polarizationlike())

# Set spectral and polarization model (here spectrum is fixed)
band = Band()

band.xp.prior = Uniform_prior(lower_bound=100, upper_bound=500)
band.xp.bounds = (None, None)
band.xp.value = 266.04

band.K.bounds = (1e-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e1)
band.K.value = 0.17

band.alpha.bounds = (-2.5, 1.0)
band.alpha.prior = Truncated_gaussian(
    mu=-0.77, sigma=0.15, lower_bound=-1.5, upper_bound=0.0
)
band.alpha.value = -0.77

band.beta.bounds = (None, -1.0)
band.beta.prior = Truncated_gaussian(
    mu=-2.67, sigma=0.15, lower_bound=-3.0, upper_bound=-1.0
)
band.beta.value = -2.67

# set up polarization model
lp = LinearPolarization(50, 90)
lp.angle.set_uninformative_prior(Uniform_prior)
lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)
lp.angle.prior = Uniform_prior(lower_bound=0.0, upper_bound=180.0)

# adding both component and defining the point source
sc = SpectralComponent("Band", band, lp)
ps = PointSource("GRB160325A", 0, 0, components=[sc])

# final model with spectrum, pol and source type info
combined_model = Model(ps)

# create the total polarization data list.
# NOTE: Can add spectral data list here if you want to do spectro-polarimetry. But right not we fix the spectrum
datalist = DataList(*datalist_daksha, polar_data)

# wrapping for 180 period of the PA
wrapped = [0] * len(combined_model.free_parameters)
wrapped[-1] = 1

# Setting up sampler and running bayes
bayes = BayesianAnalysis(combined_model, datalist)
bayes.set_sampler("multinest")
bayes.sampler.setup(
    n_live_points=500,
    resume=False,
    importance_nested_sampling=False,
    verbose=True,
    wrapped_params=wrapped,
    chain_name="chains/synch_p2",
)
bayes.sample()

# get corner plots and save them
cornerplot = bayes.results.corner_plot()
cornerplot.savefig(
    "corner_plot_POLAR_Daksha_joint-fit_GRB160325A_PF50_PA117.png", bbox_inches="tight"
)
print(
    "Corner plot saved at:",
    "corner_plot_POLAR_Daksha_joint-fit_GRB160325A_PF50_PA117.png",
)

# save the posteriors
bayes.results.write_to(
    "fit_results_POLAR_Daksha_joint-fit_GRB160325A_PF50_PA117.fits", overwrite=True
)

# modulation curves (POLAR)
modulationcurve_polar = polar_data.display()
modulationcurve_polar.savefig(
    "POLAR_modulation_curve_joint-fit_GRB160325A_PF50_PA117.png", bbox_inches="tight"
)


# modulation curves (Daksha)
for i, face in enumerate(FACES):
    modulationcurve = datalist_daksha[i].display()
    modulationcurve.savefig(
        f"Daksha_modulation_curve_face_{face:0d}_joint-fit_GRB160325A_PF50_PA117.png"
    )
