#!/usr/bin/env python3

import os
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import corner
from PIL import Image

warnings.simplefilter("ignore")
np.seterr(all="ignore")

# ---------------------------------------------------------
# Publication Formatting Setup
# ---------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
RESULTS_DIR = "results_v1"
PLOTS_DIR = "publication_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

GRB_PARAMS = {
    "GRB171010A":  {"faces": [1, 5, 12, 4, 0],   "sky_PA": 146},
    "GRB160910A":  {"faces": [6, 1, 2, 5, 7],    "sky_PA": 112},  # Updated to match your latest block
    "GRB180914B":  {"faces": [2, 0, 3, 8, 7],    "sky_PA": 34},
    "GRB180103A":  {"faces": [12, 11, 4, 5, 10], "sky_PA": 72}
}

PFS = [0.3, 0.5, 0.8]

# ---------------------------------------------------------
# Exact Filepath Generators
# ---------------------------------------------------------
def get_fits_filepath(grb_name, face_str, pf):
    """Matches: Daksha_polarization_results_GRB_X_Y_Z_PF_0.x.fits"""
    return os.path.join(RESULTS_DIR, grb_name, f"Daksha_polarization_results_{grb_name}_{face_str}_PF_{pf}.fits")

def get_png_filepath(grb_name, face_str, pf):
    """Matches: GRB_face_X_Y_Z_PF_0.x_modulation_curve_harmonic.png"""
    return os.path.join(RESULTS_DIR, grb_name, f"{grb_name}_face_{face_str}_PF_{pf}_modulation_curve_harmonic.png")

# ---------------------------------------------------------
# Posterior Data Handlers
# ---------------------------------------------------------
def load_samples(filepath):
    hdul = fits.open(filepath)
    data = hdul['ANALYSIS_RESULTS'].data
    samples = np.array(data['SAMPLES'])

    # Fix orientation
    if samples.shape[0] < samples.shape[1]:
        samples = samples.T

    samples = samples[~np.isnan(samples).any(axis=1)]
    return samples

def pa_diff(a, b):
    d = abs(a - b)
    return min(d, 180 - d)

def find_best_single_face(grb, faces, pf, true_pf, true_pa):
    best_face = None
    best_score = np.inf

    for face in faces:
        fname = get_fits_filepath(grb, str(face), pf)
        if not os.path.exists(fname): continue

        try:
            samples = load_samples(fname)
            # PF, PA ordering
            pf_samples = samples[:, 0]
            pa_samples = samples[:, 1]

            pf_mean = np.mean(pf_samples)
            pa_mean = np.mean(pa_samples)

            score = pa_diff(pa_mean, true_pa)**2 + (abs(pf_mean - true_pf) * 100)**2

            if score < best_score:
                best_score = score
                best_face = face
        except Exception:
            continue
            
    # Print omitted for standard run, but can be restored for debug
    return best_face

def get_global_ranges(grb, faces, pfs):
    all_samples = []
    for pf in pfs:
        for face in faces:
            fname = get_fits_filepath(grb, str(face), pf)
            if not os.path.exists(fname): continue
            
            try:
                samples = load_samples(fname)
                all_samples.append(samples)
            except:
                continue

    if len(all_samples) == 0: return None
    all_samples = np.vstack(all_samples)
    
    pf_vals = all_samples[:, 0]
    pa_vals = all_samples[:, 1]

    # robust limits (avoid outliers)
    pf_min, pf_max = np.percentile(pf_vals, [1, 99])
    pa_min, pa_max = np.percentile(pa_vals, [1, 99])

    return [(pf_min, pf_max), (pa_min, pa_max)]

def make_corner_image(samples, true_pf, true_pa, plot_range=None):
    if samples.shape[0] < 20: raise ValueError("Too few samples")

    fig = corner.corner(
        samples,
        labels=None, 
        levels=(0.68, 0.90, 0.95),
        plot_datapoints=False,
        fill_contours=True,
        show_titles=False,
        range=plot_range
    )

    means = np.percentile(samples, [50], axis=0)[0]
    stds  = np.std(samples, axis=0)
    ndim = samples.shape[1]

    for i, ax in enumerate(fig.axes):
        row = i // ndim
        col = i % ndim

        # Diagonal panels only (1D Histograms)
        if row == col:
            param_index = row
            mu = means[param_index]
            sigma = stds[param_index]

            ax.axvline(mu, color="black", lw=1.5, linestyle="--")
            ax.axvline(mu - sigma, color="black", lw=1.2, linestyle="--")
            ax.axvline(mu + sigma, color="black", lw=1.2, linestyle="--")

            ax.axvline(mu - 3*sigma, color="blue", lw=1.2, linestyle="--")
            ax.axvline(mu + 3*sigma, color="blue", lw=1.2, linestyle="--")

            if row == 0 and col == 0: ax.axvline(true_pf, color="red", lw=1.5)
            if row == 1 and col == 1: ax.axvline(true_pa, color="red", lw=1.5)

        # 2D panel (PA vs PF)
        if row == 1 and col == 0:
            ax.axvline(true_pf, color="red", lw=1.5)
            ax.axhline(true_pa, color="red", lw=1.5)
    
    # Add recovered values text in top right corner
    fig.text(0.95, 0.95, f"PF: {means[0]:.3f}±{stds[0]:.3f}\nPA: {means[1]:.1f}±{stds[1]:.1f}°\nTrue PF: {true_pf:.3f}\nTrue PA: {true_pa:.1f}°",
             transform=fig.transFigure, ha="right", va="top", fontsize=16,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------
# Main Plotting Routines
# ---------------------------------------------------------
def plot_posteriors_grid():
    for grb, cfg in GRB_PARAMS.items():
        print(f"\nProcessing Posteriors for {grb}...")
        plot_range = get_global_ranges(grb, cfg["faces"], PFS)

        if plot_range is None:
            print(f"  [-] No valid data for {grb}")
            continue

        try:
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))

            for i, pf in enumerate(PFS):
                faces = cfg["faces"]
                best_face = find_best_single_face(grb, faces, pf, true_pf=pf, true_pa=cfg["sky_PA"])

                combos = [
                    [best_face] if best_face is not None else [],
                    faces[:3],
                    faces
                ]

                for j, combo in enumerate(combos):
                    ax = axes[i, j]
                    
                    # ENFORCED: using underscore joining here
                    face_str = "_".join(map(str, combo))
                    
                    if not face_str:
                        ax.axis("off")
                        continue
                        
                    fname = get_fits_filepath(grb, face_str, pf)

                    if not os.path.exists(fname):
                        ax.text(0.5, 0.5, "Missing file", ha="center", va="center")
                        ax.axis("off")
                        continue

                    try:
                        samples = load_samples(fname)
                        img = make_corner_image(samples, true_pf=pf*100, true_pa=cfg["sky_PA"], plot_range=plot_range)

                        ax.imshow(img)
                        ax.axis("off")

                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error\n{str(e)}", ha="center", va="center", fontsize=8)
                        ax.axis("off")

            # Publication layout tuning
            col_titles = [f"Single ({combos[0]})", f"Triple ({combos[1]})", f"Pentuple ({combos[2]})"]
            for j, title in enumerate(col_titles):
                axes[0, j].set_title(title, fontsize=14, pad=10)

            for i, pf in enumerate(PFS):
                axes[i, 0].annotate(
                    rf"$\mathrm{{True}}\ \Pi = {pf*100}\%$", # Formatted for LaTeX
                    xy=(-0.35, 0.5),
                    xycoords="axes fraction",
                    ha="center", va="center", rotation=90, fontsize=14
                )
                
            fig.text(0.5, 0.04, r"$\mathrm{Polarisation\ Fraction\ (\Pi)}$", ha="center", fontsize=16)
            fig.text(0.04, 0.5, r"$\mathrm{Sky\ Polarisation\ Angle\ (\phi)}$", va="center", rotation="vertical", fontsize=16)
            fig.suptitle(rf"\textbf{{{grb} Joint Posteriors}}", fontsize=18, y=0.98)

            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92, wspace=0.00, hspace=0.05)

            out = os.path.join(PLOTS_DIR, f"{grb}_Posterior_Grid.pdf") # PDF for vector scaling
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  [+] Saved: {out}")

        except Exception as e:
            print(f"  [X] Failed GRB {grb}: {e}")
            continue

def plot_modulation_image_grid():
    for grb, cfg in GRB_PARAMS.items():
        print(f"\nStitching Modulation Grid for {grb}...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(rf"\textbf{{{grb} Modulation Curves}}", fontsize=20, y=0.98)
        
        for row_idx, pf in enumerate(PFS):
            faces = cfg["faces"]
            best_face = find_best_single_face(grb, faces, pf, true_pf=pf, true_pa=cfg["sky_PA"])
            
            combos = [
                [best_face] if best_face is not None else [],
                faces[:3],
                faces
            ]
            
            for col_idx, combo in enumerate(combos):
                ax = axes[row_idx, col_idx]
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                
                # ENFORCED: using underscore joining here
                face_str = "_".join(map(str, combo))
                
                # Set Titles matching the Posterior grid
                if row_idx == 0: 
                    titles = ["Single", "Triple", "Pentuple"]
                    ax.set_title(titles[col_idx], pad=10, fontsize=14)
                    
                if col_idx == 0:
                    ax.set_ylabel(rf"$\mathrm{{True}}\ \Pi = {pf*100}\%$", fontsize=14, labelpad=20)
                    ax.spines['left'].set_visible(True)
                    ax.spines['left'].set_color('none')
                    
                if not face_str:
                    ax.text(0.5, 0.5, "Data Not Applicable", ha='center', va='center', fontsize=12, color='gray')
                    continue
                
                png_path = get_png_filepath(grb, face_str, pf)
                
                if os.path.exists(png_path):
                    img = mpimg.imread(png_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "Image File\nNot Found", ha='center', va='center', fontsize=12, color='red')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
        out = os.path.join(PLOTS_DIR, f"{grb}_Modulation_Grid.pdf")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [+] Saved: {out}")

if __name__ == "__main__":
    plot_posteriors_grid()
    plot_modulation_image_grid()
    print("\n✅ Done — all GRBs processed")