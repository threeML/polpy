import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def extract_prsp_data(response_file, pa_offset=0.0):
    """Extracts and sorts the relevant polarization matrices and energies from the FITS file."""
    with fits.open(response_file) as hdu_pol:
        ene_lo = np.array(hdu_pol['INEBOUNDS'].data.field('ENERG_LO'), dtype=np.float64)
        ene_hi = np.array(hdu_pol['INEBOUNDS'].data.field('ENERG_HI'), dtype=np.float64)
        energy_mid = (ene_lo + ene_hi) / 2.

        pol_ang = np.array(hdu_pol['INPAVALS'].data.field('PA_IN'), dtype=np.float64)
        pol_ang = (180 + pol_ang - pa_offset) % 180
        
        sorted_indices = np.argsort(pol_ang)
        pol_ang_sorted = pol_ang[sorted_indices]

        polmatrix = hdu_pol['SPECRESP POLMATRIX'].data
        polmatrix = polmatrix[:, sorted_indices, :]
        polmatrix = polmatrix.transpose() # Shape becomes: (N_E, N_PA, N_SA)

    return energy_mid, pol_ang_sorted, polmatrix

def compute_harmonic_fit(pa_train_deg, y_train, pa_eval_deg, n_harmonics=2):
    """Fits harmonics to training data and evaluates at specified points."""
    pa_train_rad = np.deg2rad(pa_train_deg)
    pa_eval_rad = np.deg2rad(pa_eval_deg)

    X_train = np.ones((len(pa_train_rad), 1))
    X_eval = np.ones((len(pa_eval_rad), 1))

    for i in range(1, n_harmonics + 1):
        X_train = np.hstack([X_train, np.cos(2 * i * pa_train_rad[:, None]), np.sin(2 * i * pa_train_rad[:, None])])
        X_eval = np.hstack([X_eval, np.cos(2 * i * pa_eval_rad[:, None]), np.sin(2 * i * pa_eval_rad[:, None])])

    w, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    
    return X_eval @ w

def add_stats_box(ax, data):
    """Helper function to add a text box with mean and std dev to an axis."""
    if len(data) > 0:
        mu = np.mean(data)
        sigma = np.std(data)
        textstr = f"$\\mu = {mu:.3f}\\%$\n$\\sigma = {sigma:.3f}\\%$"
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

def main(grb, pa, pf, face):
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    RESPONSE_FILE = f'/home/polpy/daksha_data/{grb}_response/face_{face}/DAKSHA_POLRSP_EMIN_100_EMAX_1000_{grb}_{face}.prsp'
    PA_OFFSET = 0.0
    ENERGY_SPLIT = 700.0 # keV threshold
    
    try:
        energies, pa_raw, polmatrix = extract_prsp_data(RESPONSE_FILE, PA_OFFSET)
    except FileNotFoundError:
        print(f"Error: Could not find {RESPONSE_FILE}.")
        return

    N_E, N_PA, N_SA = polmatrix.shape
    print(f"Loaded {N_PA} Polarization Angles. Processing residuals for 2 vs 3 harmonics...")

    # Lists to store our residual percentages
    res_2h_low = []
    res_3h_low = []
    res_2h_high = []
    res_3h_high = []

    # ==========================================
    # 2. COMPUTE RESIDUALS ACROSS ALL BINS
    # ==========================================
    for e in range(N_E):
        current_energy = energies[e]
        for sa in range(N_SA):
            y_raw = polmatrix[e, :, sa]

            # Compute both 2-harmonic and 3-harmonic fits evaluated at the original PAs
            y_pred_2h = compute_harmonic_fit(pa_raw, y_raw, pa_raw, n_harmonics=2)
            y_pred_3h = compute_harmonic_fit(pa_raw, y_raw, pa_raw, n_harmonics=3)
            
            for i in range(N_PA):
                if y_raw[i] != 0:
                    err_pct_2h = ((y_pred_2h[i] - y_raw[i]) / y_raw[i]) * 100
                    err_pct_3h = ((y_pred_3h[i] - y_raw[i]) / y_raw[i]) * 100
                    
                    if current_energy <= ENERGY_SPLIT:
                        res_2h_low.append(err_pct_2h)
                        res_3h_low.append(err_pct_3h)
                    else:
                        res_2h_high.append(err_pct_2h)
                        res_3h_high.append(err_pct_3h)

    # ==========================================
    # 3. PLOT HISTOGRAMS (2x2 GRID)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.linspace(-100, 100, 300)

    # --- Row 1: Energies <= 400 keV ---
    axes[0, 0].hist(res_2h_low, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f"2-Harmonic Fit Residuals\nE $\leq$ {ENERGY_SPLIT} keV")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[0, 0], res_2h_low)

    axes[0, 1].hist(res_3h_low, bins=bins, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title(f"3-Harmonic Fit Residuals\nE $\leq$ {ENERGY_SPLIT} keV")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[0, 1], res_3h_low)

    # --- Row 2: Energies > 400 keV ---
    axes[1, 0].hist(res_2h_high, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title(f"2-Harmonic Fit Residuals\nE > {ENERGY_SPLIT} keV")
    axes[1, 0].set_xlabel("Residual Percentage (%) [(Pred - True) / True]")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[1, 0], res_2h_high)

    axes[1, 1].hist(res_3h_high, bins=bins, color='firebrick', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title(f"3-Harmonic Fit Residuals\nE > {ENERGY_SPLIT} keV")
    axes[1, 1].set_xlabel("Residual Percentage (%) [(Pred - True) / True]")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[1, 1], res_3h_high)

    plt.tight_layout()
    
    # Save as PDF
    out_filename = f'{grb}_PA_{pa}_PF_{pf}_face_{face}_2v3_harmonic_residuals.pdf'
    plt.savefig(out_filename, format='pdf', bbox_inches='tight')
    
    print("\n--- Summary Statistics ---")
    print(f"Low Energy (<= {ENERGY_SPLIT} keV):")
    print(f"  2-Harmonic Std Dev: {np.std(res_2h_low):.3f}%")
    print(f"  3-Harmonic Std Dev: {np.std(res_3h_low):.3f}%")
    print(f"High Energy (> {ENERGY_SPLIT} keV):")
    print(f"  2-Harmonic Std Dev: {np.std(res_2h_high):.3f}%")
    print(f"  3-Harmonic Std Dev: {np.std(res_3h_high):.3f}%")
    print(f"\nPlot saved successfully to: {out_filename}")

if __name__ == "__main__":
    grb = "GRB180914B_templates_20M10M"
    pa = "068"
    pf = 0.5
    face = 0
    main(grb, pa, pf, face)