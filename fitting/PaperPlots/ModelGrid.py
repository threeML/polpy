#!/usr/bin/env python3  

import itertools
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
np.seterr(all="ignore")

from threeML import *
silence_warnings()
set_threeML_style()

# ---------------------------------------------------------
# Configuration & Inputs
# ---------------------------------------------------------
DATA_DIR = "/home/polpy/daksha_data"
TRIGGER_TIME = 0

GRB_PARAMS = {
    "GRB180103A":  {"faces": [12, 11, 4, 5, 10], "local_PA": 122, "sky_PA": 72, "Norm": 0.028,  "alpha": -1.31, "beta": -2.24, "Epeak": 273.00, "t_src": 165.83, "t_bkg": 100.0},
    "GRB171010A":  {"faces": [1, 5, 12, 4, 0],  "local_PA": 40, "sky_PA": 146, "Norm": 0.116,  "alpha": -0.91, "beta": -2.28, "Epeak": 228.77, "t_src": 104.0,  "t_bkg": 100.0},
    "GRB160910A":  {"faces": [6, 1, 2, 5, 7],   "local_PA": 46,  "sky_PA": 45, "Norm": 0.020,  "alpha": -0.81, "beta": -2.17, "Epeak": 321.90, "t_src": 11.0,   "t_bkg": 100.0},
    "GRB180914B":  {"faces": [2, 0, 3, 8, 7],   "local_PA": 68,  "sky_PA": 34, "Norm": 0.0555, "alpha": -0.75, "beta": -2.10, "Epeak": 453.00, "t_src": 160.0,  "t_bkg": 100.0}
    }

PFS_TO_RUN = [0.3, 0.5, 0.8]

def run_polarization_analysis(grb_name, params, pf, current_faces):
    """
    Runs the Bayesian polarization analysis for a specific GRB, PF, and set of faces.
    """
    faces_str = ''.join(map(str, current_faces))
    pa_str = f"{params['local_PA']:03d}" # Format with leading zeros (e.g., 68 -> "068")
    
    print(f"\n{'='*50}")
    print(f"Running {grb_name} | PF: {pf} | Faces: {faces_str}")
    print(f"{'='*50}")

    datalist_items = []
    
    # Calculate intervals based on t_src and t_bkg
    # Assuming the source is centered at 0 and there is a 10s gap before background
    t_src_half = params["t_src"] / 2.0
    # t_bkg = params["t_bkg"]
    # gap = 10.0
    
    active_interval = f"{-t_src_half} - {t_src_half}"
    bkg_interval_1 = f"-245 - -100" # Using fixed background intervals as in original code
    bkg_interval_2 = f"100 - 245"
    if grb_name == "GRB180103A": # Adjusting background intervals for GRB180103A as per original code
        bkg_interval_1 = f"-205 - -90"
        bkg_interval_2 = f"90 - 200"

    # Load data for each face in the current combination
    for face in current_faces:
        pevt_path = f"{DATA_DIR}/{grb_name}_PF_{pf}_1.0x/{grb_name}_PA_{pa_str}_PF_{pf}_face_{face}_1.0x_dakshapol.pevt"
        if grb_name == "GRB180914B": #there is a 20M10M response for this GRB, so we use that instead of the regular response
            prsp_path = f"{DATA_DIR}/{grb_name}_templates_20M10M_response/face_{face}/DAKSHA_POLRSP_EMIN_100_EMAX_1000_{grb_name}_templates_20M10M_{face}.prsp"
        else:
            prsp_path = f"{DATA_DIR}/{grb_name}_response/face_{face}/DAKSHA_POLRSP_EMIN_100_EMAX_1000_{grb_name}_{face}.prsp"

        ts = TimeSeriesBuilder.from_polarization(
            name=f'daksha_pol{face}',
            polevents=pevt_path,
            polrsp=prsp_path,
            specrsp=None,
            trigger_time=TRIGGER_TIME
        )
        
        ts.set_active_time_interval(active_interval)
        ts.set_background_interval(bkg_interval_1, bkg_interval_2)
        datalist_items.append(ts.to_polarizationlike())

    # Setting up spectrum model
    band = Band()
    
    # Assign values from dictionary and reset bounds to prevent boundary errors
    band.xp.bounds = (None, None)
    band.xp.value = params["Epeak"]
    
    band.K.bounds = (1E-10, None)
    band.K.value = params["Norm"]
    
    band.alpha.bounds = (-3.0, 2.0)
    band.alpha.value = params["alpha"]
    
    band.beta.bounds = (None, -1.0)
    band.beta.value = params["beta"]

    # Setting up polarization model
    lp = LinearPolarization(50, 90)
    lp.angle.prior = Uniform_prior(lower_bound=0.0, upper_bound=180.0)
    lp.degree.prior = Uniform_prior(lower_bound=0.001, upper_bound=100.0)

    # Adding both component and defining the point source
    sc = SpectralComponent('synch', band, lp)
    ps = PointSource(grb_name, 0, 0, components=[sc])

    combined_model = Model(ps)
    datalist = DataList(*datalist_items)

    # Freeze spectral parameters
    combined_model[grb_name].spectrum.synch.Band.K.free = False
    combined_model[grb_name].spectrum.synch.Band.alpha.free = False
    combined_model[grb_name].spectrum.synch.Band.beta.free = False
    combined_model[grb_name].spectrum.synch.Band.xp.free = False

    # Bayesian analysis
    chain_name = os.path.join(CHAINS_DIR, f"synch_{grb_name}_faces{'_'.join(map(str, current_faces))}_PF{pf}")
    bayes = BayesianAnalysis(combined_model, datalist)
    bayes.set_sampler("multinest")
    
    wrapped = [0] * len(combined_model.free_parameters)
    bayes.sampler.setup(
        n_live_points=1000, 
        resume=False, 
        importance_nested_sampling=False,
        verbose=True, 
        wrapped_params=wrapped, 
        chain_name=chain_name
    )
    bayes.sample()

    # Save and display results
    fits_filename = os.path.join(RESULTS_DIR, f"Daksha_polarization_results_{grb_name}_{'_'.join(map(str, current_faces))}_PF_{pf}.fits")
    bayes.results.write_to(fits_filename, overwrite=True)
    
    bayes.restore_median_fit()
    
    # Corner plot
    cornerplot = bayes.results.corner_plot()
    corner_filename = os.path.join(RESULTS_DIR, f"{grb_name}_face_{'_'.join(map(str, current_faces))}_PF_{pf}_corner_plot_harmonic.png")
    cornerplot.savefig(corner_filename)
    print(f"Corner plot saved at: {corner_filename}")
    plt.close(cornerplot) # clean up

    # Modulation curves
    for i, face in enumerate(current_faces):
        modulationcurve = datalist_items[i].display()
        mod_filename = os.path.join(RESULTS_DIR, f"{grb_name}_face_{'_'.join(map(str, current_faces))}_PF_{pf}_modulation_curve_harmonic.png")
        modulationcurve.savefig(mod_filename)
        plt.close(modulationcurve) # clean up
        
    plt.close('all')

# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
for grb_name, params in GRB_PARAMS.items():
    faces_list = params["faces"]

    RESULTS_DIR = os.path.join("results_v1", grb_name)  # Update RESULTS_DIR to subdirectory for current GRB 
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists for current GRB
    CHAINS_DIR = os.path.join(RESULTS_DIR, "chains")  # Update CHAINS_DIR to subdirectory for current GRB
    os.makedirs(CHAINS_DIR, exist_ok=True)  # Ensure chains directory exists for current GRB
    
    # Generate the combinations: Individual faces, triplets, and quintuplets
    face_combinations = [[faces_list[i]] for i in range(len(faces_list))]  # Individual faces
    face_combinations += [faces_list[0:3]]
    face_combinations += [faces_list]
    for pf in PFS_TO_RUN:
        for current_faces in face_combinations:
            try:
                run_polarization_analysis(grb_name, params, pf, current_faces)
            except Exception as e:
                print(f"Failed to process {grb_name} | PF: {pf} | Faces: {current_faces}.")
                print(f"Error: {e}")
                continue

print("\nAll runs complete! Check the 'results/' directory.")