import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
import argparse

def extract_prsp_data(response_file, pa_offset=0.0):
    """Extracts and sorts the relevant polarization matrices from the FITS file."""
    with fits.open(response_file) as hdu_pol:
        ene_lo = np.array(hdu_pol['INEBOUNDS'].data.field('ENERG_LO'), dtype=np.float64)
        ene_hi = np.array(hdu_pol['INEBOUNDS'].data.field('ENERG_HI'), dtype=np.float64)
        energy_mid = (ene_lo + ene_hi) / 2.

        pol_ang = np.array(hdu_pol['INPAVALS'].data.field('PA_IN'), dtype=np.float64)
        pol_ang = (180 + pol_ang - pa_offset) % 180
        
        sorted_indices = np.argsort(pol_ang)
        pol_ang_sorted = pol_ang[sorted_indices]

        samin = np.array(hdu_pol['SABOUNDS'].data.field('SA_MIN'), dtype=np.float64)
        samax = np.array(hdu_pol['SABOUNDS'].data.field('SA_MAX'), dtype=np.float64)
        bins = np.append(samin, samax[-1])
        sa_mid = 0.5 * (bins[:-1] + bins[1:])

        polmatrix = hdu_pol['SPECRESP POLMATRIX'].data
        polmatrix = polmatrix[:, sorted_indices, :]
        polmatrix = polmatrix.transpose() # Shape: (N_E, N_PA, N_SA)

    return energy_mid, pol_ang_sorted, sa_mid, polmatrix

def compute_harmonic_fit(pa_deg, y_vals, fine_pa_deg, n_harmonics):
    """Computes the least squares harmonic fit."""
    pa_rad = np.deg2rad(pa_deg)
    fine_pa_rad = np.deg2rad(fine_pa_deg)

    X = np.ones((len(pa_rad), 1))
    X_fine = np.ones((len(fine_pa_rad), 1))

    for i in range(1, n_harmonics + 1):
        X = np.hstack([X, np.cos(2 * i * pa_rad[:, None]), np.sin(2 * i * pa_rad[:, None])])
        X_fine = np.hstack([X_fine, np.cos(2 * i * fine_pa_rad[:, None]), np.sin(2 * i * fine_pa_rad[:, None])])

    w, _, _, _ = np.linalg.lstsq(X, y_vals, rcond=None)
    return X_fine @ w

def get_closest_indices(array, targets):
    """Finds the indices of the array that are closest to the target values."""
    indices = []
    for t in targets:
        idx = (np.abs(array - t)).argmin()
        indices.append(idx)
    return indices

def main(grb, face):
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    TARGET_ENERGIES = [200, 300, 350, 400, 500, 700] # keV
    PA_OFFSET = 0.0
    output_pdf = f'{grb}_face_{face}_interpolation_plots.pdf'

    energies, pa_raw, sa_bins, polmatrix = extract_prsp_data(f'/home/polpy/daksha_data/{grb}_response/face_{face}/DAKSHA_POLRSP_EMIN_100_EMAX_1000_{grb}_{face}.prsp', PA_OFFSET)

    # Find the actual FITS energy bins closest to our targets
    e_indices = get_closest_indices(energies, TARGET_ENERGIES)
    
    print("Using closest available FITS energy bins:")
    for i, target in enumerate(TARGET_ENERGIES):
        print(f"  Target: {target} keV --> Actual Bin: {energies[e_indices[i]]:.2f} keV (Index {e_indices[i]})")

    fine_pa = np.linspace(0, 180, 1810) # 0.1 degree resolution for smooth curves

    # ---------------------------------------------------------
    # PDF Generation Loop
    # ---------------------------------------------------------
    print(f"Generating PDF with {len(sa_bins)} pages...")
    
    with PdfPages(output_pdf) as pdf:
        # Loop through every single scattering angle bin
        for sa_idx, sa_val in enumerate(sa_bins):
            
            # Create a 2x3 grid for the 6 energies
            fig, axes = plt.subplots(2, 3, figsize=(14, 10))
            fig.suptitle(f"{grb} - Face {face} Scattering Angle Bin: {sa_val:.2f}° (Index: {sa_idx})", fontsize=16, fontweight='bold')

            # Flatten the 2x3 axes array to easily loop over it
            for i, ax in enumerate(axes.flatten()):
                e_idx = e_indices[i]
                e_val = energies[e_idx]
                target_val = TARGET_ENERGIES[i]

                # Extract data
                y_raw = polmatrix[e_idx, :, sa_idx]
                
                # Compute fits
                y_fit_2 = compute_harmonic_fit(pa_raw, y_raw, fine_pa, n_harmonics=2)
                y_fit_3 = compute_harmonic_fit(pa_raw, y_raw, fine_pa, n_harmonics=3)

                # Plotting
                ax.plot(pa_raw, y_raw, 'ko-', label='Raw Data (Linear)', markersize=5, linewidth=1.5, alpha=0.7)
                ax.plot(fine_pa, y_fit_2, 'b-', label='2-Harmonic Fit', linewidth=2)
                ax.plot(fine_pa, y_fit_3, 'r--', label='3-Harmonic Fit', linewidth=2)

                ax.set_title(f"Energy: ~{e_val:.1f} keV (Target: {target_val})")
                ax.set_xlabel("Polarization Angle (Degrees)")
                ax.set_ylabel("Response / Counts")
                ax.set_xlim(0, 180)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # Only put the legend on the first plot to save space
                if i == 0:
                    ax.legend()

            # Adjust layout so titles don't overlap
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure to the current page of the PDF
            pdf.savefig(fig)
            
            # Close the figure to free up memory (CRITICAL for large loops)
            plt.close(fig)

    print(f"Done! Successfully saved to {output_pdf}")

if __name__ == "__main__":
    main("GRB180914B_templates_20M10M", 0)
    main("GRB180914B_templates_20M10M", 2)
    main("GRB180914B_templates_20M10M", 3)
    main("GRB180914B_templates_20M10M", 4)
    main("GRB180914B_templates_20M10M", 10)
    main("GRB180914B_templates_20M10M", 11)
    main("GRB180914B_templates_20M10M", 12)

