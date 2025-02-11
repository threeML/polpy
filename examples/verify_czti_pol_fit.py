
%matplotlib tk

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.integrate import simpson
from scipy.interpolate import interp1d


def band_photspec(E, norm, Epeak, alpha, beta):
    E0 = Epeak / (2 + alpha)

    fE = np.zeros(E.size)

    mask = E < (alpha - beta) * E0
    fE[mask] = norm * (E[mask] / 100)**alpha * np.exp(-(E[mask] / E0))
    fE[~mask] = norm * ((alpha - beta) * E0 / 100)**(alpha - beta) * np.exp(beta - alpha) * (E[~mask]/100)**beta

    return fE


# grbname
grbname = "GRB180120A"

# read data and response
hist1 = np.loadtxt(f"/home/sujay/local/data/3ml-test/czti_pol_fit_verification/source_hist_{grbname:s}.txt", delimiter=',')
hist2 = np.loadtxt("/home/sujay/local/data/3ml-test/czti_pol_fit_verification/source_hist_GRB180427A.txt", delimiter=',')
hist3 = np.loadtxt("/home/sujay/local/data/3ml-test/czti_pol_fit_verification/source_hist_GRB190530A.txt", delimiter=',')

rspHDU = fits.open(f"/home/sujay/local/data/3ml-test/czti_pol_fit_verification/CZTI_POLRSP_{grbname:s}.prsp")

# pol and spectral params
fit_paramHDU = fits.open(f"/home/sujay/local/data/3ml-test/czti_pol_fit_verification/AstroSat_CZTI_polarization_results_{grbname:s}.fits")
norm = fit_paramHDU[1].data['Value'][0]
alpha = fit_paramHDU[1].data['Value'][1]
epeak = fit_paramHDU[1].data['Value'][2]
beta = fit_paramHDU[1].data['Value'][3]
t90 = 26.9
print(norm, alpha, beta, epeak)

esim = ((rspHDU[1].data['ENERG_LO'] + rspHDU[1].data['ENERG_HI']) / 2)
pasim = rspHDU[2].data['PA_IN']
pol_matrix = rspHDU[4].data.T
unpol_matrix = rspHDU[5].data.T

# convolve with GRB spectrum
weights = band_photspec(esim, norm, epeak, alpha, beta)  # ph/s/cm2/keV
pol_hist_grb_scaled = simpson(pol_matrix * weights[:, None, None], esim, axis=0) # cnt/s
unpol_hist_grb_scaled = simpson(unpol_matrix * weights[:, None], esim, axis=0) # cnt/s
pa = fit_paramHDU[1].data['Value'][5]
pf = fit_paramHDU[1].data['Value'][4]/100

pol_hist_interp_scaled = interp1d(pasim, pol_hist_grb_scaled, axis=0, kind='cubic', fill_value='extrapolate')
pol_hist_grb_scaled_val = pol_hist_interp_scaled(pa)

hist_model_scaled = pol_hist_grb_scaled_val * pf + (1 - pf)*unpol_hist_grb_scaled

# read direct band sim histograms
direct_hist_path = Path(f"/home/sujay/local/data/3ml-test/czti_pol_fit_verification/old_pipeline_mu_0_100/{grbname:s}/")

unpol_hist_grb_direct = np.loadtxt(direct_hist_path.joinpath(f"unpol/hist_EventFileSimul{grbname[:-1]:s}_mass.txt.ver2.dat_100_600_thr_20.txt"))
pol_hist_grb_direct = np.zeros((18, 8))

for i, fname in enumerate(sorted(direct_hist_path.glob("*.txt"))[:-1]):
    pol_hist_grb_direct[i, :] = np.loadtxt(fname)
    

pol_hist_interp_direct = interp1d(pasim, pol_hist_grb_scaled, axis=0, kind='cubic', fill_value='extrapolate')
pol_hist_grb_direct_val = pol_hist_interp_scaled(pa)
hist_model_direct = pol_hist_grb_direct_val * pf + (1 - pf)*unpol_hist_grb_direct

# plot scaled and unscaled
scat_bins = np.arange(0, 360, 45)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(scat_bins, unpol_hist_grb_direct/unpol_hist_grb_direct.mean(), ds="steps-mid", label="Direct")
ax.plot(scat_bins, unpol_hist_grb_scaled/unpol_hist_grb_scaled.mean(), ds="steps-mid", label="Scaled")
ax.legend()
fig.suptitle("Unpolarized histogram comparison")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(scat_bins, hist1[:, 1]/hist1[:, 1].mean(), yerr = hist1[:, 2]/hist1[:, 1].mean(), fmt='o', label=f"{grbname:s}, $\\theta = 15.89^\circ$")
#ax.errorbar(scat_bins, hist2[:, 1]/hist2[:, 1].mean(), yerr = hist2[:, 2]/hist2[:, 1].mean(), fmt='o', label='GRB108427A, $\\theta = 40.81^\circ$')
#ax.errorbar(scat_bins, hist3[:, 1]/hist3[:, 1].mean(), yerr = hist3[:, 2]/hist3[:, 1].mean(), fmt='o', label='GRB190530A, $\\theta = 154.5^\circ$')
plt.legend()
ax.plot(scat_bins, hist_model_scaled/hist_model_scaled.mean(), ds='steps-mid', label="Scaled")
ax.plot(scat_bins, hist_model_direct/hist_model_direct.mean(), ds='steps-mid', label="Direct")
ax.legend()


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(scat_bins, hist1[:, 1]/hist1[:, 1].mean(), yerr = hist1[:, 2]/hist1[:, 1].mean(), fmt='o', label=f"{grbname:s}, $\\theta = 15.89^\circ$")
ax.errorbar(scat_bins, hist2[:, 1]/hist2[:, 1].mean(), yerr = hist2[:, 2]/hist2[:, 1].mean(), fmt='o', label='GRB108427A, $\\theta = 40.81^\circ$')
ax.errorbar(scat_bins, hist3[:, 1]/hist3[:, 1].mean(), yerr = hist3[:, 2]/hist3[:, 1].mean(), fmt='o', label='GRB190530A, $\\theta = 154.5^\circ$')
plt.legend()





