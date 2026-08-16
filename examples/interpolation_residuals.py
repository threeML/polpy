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
        
        # Place text box in the upper right corner
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

def main(grb, face):
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    RESPONSE_FILE = f'/home/polpy/daksha_data/{grb}_response/face_{face}/DAKSHA_POLRSP_EMIN_100_EMAX_1000_{grb}_{face}.prsp'
    PA_OFFSET = 0.0
    DROP_INDEX = 5 # The 5th PA value (0-indexed)
    N_HARMONICS = 2
    ENERGY_SPLIT = 400.0 # keV threshold
    
    try:
        energies, pa_raw, polmatrix = extract_prsp_data(RESPONSE_FILE, PA_OFFSET)
    except FileNotFoundError:
        print(f"Error: Could not find {RESPONSE_FILE}.")
        return

    N_E, N_PA, N_SA = polmatrix.shape
    target_pa = np.array([pa_raw[DROP_INDEX]])
    pa_train = np.delete(pa_raw, DROP_INDEX)

    print(f"Total PAs: {N_PA}")
    print(f"Dropping PA Index {DROP_INDEX} (Value: {target_pa[0]:.2f} deg) for cross-validation...")

    loocv_errors_low = []
    full_fit_errors_low = []
    loocv_errors_high = []
    full_fit_errors_high = []

    # ==========================================
    # 2. COMPUTE ERRORS ACROSS ALL BINS
    # ==========================================
    for e in range(N_E):
        current_energy = energies[e]
        for sa in range(N_SA):
            y_raw = polmatrix[e, :, sa]

            # --- A. Leave-One-Out Prediction ---
            y_train = np.delete(y_raw, DROP_INDEX)
            y_true = y_raw[DROP_INDEX]
            
            y_pred_loocv = compute_harmonic_fit(pa_train, y_train, target_pa, N_HARMONICS)[0]
            
            if y_true != 0:
                loocv_err_pct = ((y_pred_loocv - y_true) / y_true) * 100
                if current_energy <= ENERGY_SPLIT:
                    loocv_errors_low.append(loocv_err_pct)
                else:
                    loocv_errors_high.append(loocv_err_pct)

            # --- B. Full Fit Residuals ---
            y_pred_full = compute_harmonic_fit(pa_raw, y_raw, pa_raw, N_HARMONICS)
            
            for i in range(N_PA):
                if y_raw[i] != 0:
                    full_err_pct = ((y_pred_full[i] - y_raw[i]) / y_raw[i]) * 100
                    if current_energy <= ENERGY_SPLIT:
                        full_fit_errors_low.append(full_err_pct)
                    else:
                        full_fit_errors_high.append(full_err_pct)

    # ==========================================
    # 3. PLOT HISTOGRAMS (2x2 GRID)
    # ==========================================
    def add_90pct_lines(ax, data, color='green'):
        """Add vertical lines at the 5th and 95th percentiles (90% interval)."""
        if len(data) > 0:
            lower = np.percentile(data, 5)
            upper = np.percentile(data, 95)
            ax.axvline(lower, color=color, linestyle='--', linewidth=2, label='5th/95th pct')
            ax.axvline(upper, color=color, linestyle='--', linewidth=2)
            ax.legend(loc='upper left', fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bins = np.linspace(-100, 100, 300)

    # Row 1: Energies <= 400 keV
    axes[0, 0].hist(loocv_errors_low, bins=bins, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f"Predictive Error (Dropped PA: {target_pa[0]:.1f}°)\nE $\leq$ {ENERGY_SPLIT} keV -- {grb} Face {face}")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[0, 0], loocv_errors_low)
    add_90pct_lines(axes[0, 0], loocv_errors_low)

    axes[0, 1].hist(full_fit_errors_low, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title(f"General Fit Residuals\nE $\leq$ {ENERGY_SPLIT} keV -- {grb} Face {face}")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[0, 1], full_fit_errors_low)
    add_90pct_lines(axes[0, 1], full_fit_errors_low)

    # Row 2: Energies > 400 keV
    axes[1, 0].hist(loocv_errors_high, bins=bins, color='firebrick', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title(f"Predictive Error (Dropped PA: {target_pa[0]:.1f}°)\nE > {ENERGY_SPLIT} keV -- {grb} Face {face}")
    axes[1, 0].set_xlabel("Error Percentage (%) [(Pred - True) / True]")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[1, 0], loocv_errors_high)
    add_90pct_lines(axes[1, 0], loocv_errors_high)

    axes[1, 1].hist(full_fit_errors_high, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title(f"General Fit Residuals\nE > {ENERGY_SPLIT} keV -- {grb} Face {face}")
    axes[1, 1].set_xlabel("Residual Percentage (%) [(Pred - True) / True]")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(0, color='black', linestyle='dashed', linewidth=1.5)
    add_stats_box(axes[1, 1], full_fit_errors_high)
    add_90pct_lines(axes[1, 1], full_fit_errors_high)
    plt.tight_layout()
    
    # Save as PDF
    out_filename = f'{grb}_dropped_PA_{DROP_INDEX}_face_{face}_harmonic_fit_errors.pdf'
    plt.savefig(out_filename, format='pdf', bbox_inches='tight')
    print(f"\nPlot saved successfully to: {out_filename}")

if __name__ == "__main__":
    grb = "GRB180914B_templates_20M10M"
    face = 0
    main(grb, 0)
    main(grb, 1)
    main(grb, 2)
    main(grb, 3)
    main(grb, 4)
    main(grb, 10)
    main(grb, 11)
    main(grb, 12)