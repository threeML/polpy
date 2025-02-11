
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import yaml
from pathlib import Path
from threeML import *
import numpy as np


with open('/home/polpy/polpy_test/polpy/fitting/GRB170114A/GRBconfig_PA_validation.yml', 'r') as file:
    grb_config = yaml.safe_load(file)

import requests as reqs
from bs4 import BeautifulSoup as bso
import os
"""
for i in range(0,2):
# Loop through each GRB in the YAML file
    for grb_name, grb_data in grb_config.items():
        print(f"Processing {grb_name}...")

        # POLAR data and responses
        data_path = Path('/home/polpy/polpy_test/polpy/fitting').parent.joinpath('data/POLAR_14_GRB_DATA_OLD')
        polevents = data_path.joinpath(f"POLAR_{grb_name}.pevt")
        polrsp = data_path.joinpath(f"POLAR_{grb_name}.prsp")
        specrsp = data_path.joinpath(f"POLAR_{grb_name}.rmfarf")

        trigger_time = grb_data['trigger_time']

        # spectral
        polar = TimeSeriesBuilder.from_pol_spectrum('polar_spec', polevents.as_posix(), specrsp.as_posix(),
                                                    trigger_time=trigger_time)

        polar.set_active_time_interval(grb_data['polar_src_interval'][0])
        polar.set_background_interval(*grb_data['polar_bkg_intervals'])
        polar.create_time_bins(start=-100.0, stop=100.0, method='constant', dt=0.1)
        polar.view_lightcurve(-100, 100).savefig(f'/home/polpy/polpy_test/polpy/fitting/verification/{grb_name}/{grb_name}_old_lightcurve.png')

        polar_spec = polar.to_spectrumlike()
        polar_spec.set_active_measurements(grb_data['polar_active_interval'][0])
        # polarisation 
        polar_polarization_ts = TimeSeriesBuilder.from_polarization('polar_pol', polevents.as_posix(), polrsp.as_posix(),
                                                    specrsp, trigger_time=trigger_time)

        polar_polarization_ts.set_background_interval(*grb_data['polar_bkg_intervals'])
        polar_polarization_ts.set_active_time_interval(grb_data['polar_src_interval'][0])
        polar_data = polar_polarization_ts.to_polarizationlike()

        fig = polar_data.display(show_model=False, min_rate=10)
        fig.show()

        #save the figure in a folder named after the GRB
        fig.savefig(f'./{grb_name}_oldrsp_polarization.png')

        #moving to spectrum retrieval from co-detections

        specs={}
        detectors = {}

        # Check if 'BAT' is in joint detectors
        if 'BAT' in grb_data['joint_detectors']:

            #set to effective area correction mode
            polar_spec.use_effective_area_correction(grb_data['constspec_lo'], grb_data['constspec_hi'])
            polar_data.use_effective_area_correction(grb_data['constpol_lo'], grb_data['constpol_hi'])

            swift_trigger_number=grb_data['swift_file'][2:13]
            print(swift_trigger_number)
            base_url = f"https://swift.gsfc.nasa.gov/results/batgrbcat/GRB{grb_name}/data_product/{swift_trigger_number}-results/pha/"
            site = reqs.get(base_url)
            mysoup = bso(site.text, 'html.parser')
            pha_filename = f"sw{swift_trigger_number}b_t90.pha"
            rsp_filename = f"sw{swift_trigger_number}b_preslew.rsp"
            # Create directory for the GRB if it doesn't exist
            grb_directory = f"/home/polpy/polpy_test/polpy/data/{grb_name}/swiftBATdata"
            os.makedirs(grb_directory, exist_ok=True)

            # Get the HTML content of the page
            response = reqs.get(base_url)
            response.raise_for_status()
            soup = bso(response.text, 'html.parser')

            # Find all <a> tags with href attributes
            links = soup.find_all('a', href=True)

            # Download only files with .pha or .rsp extensions
            for link in links:
                href = link['href']
                # Filter for files with .pha or .rsp extensions
                if href.endswith(('.pha', '.rsp')):
                    file_url = base_url + href
                    file_path = os.path.join(grb_directory, href)
                    
                    print(f"Downloading {file_url}...")
                    file_response = reqs.get(file_url)
                    file_response.raise_for_status()  # Raise an error for bad responses
                    with open(file_path, 'wb') as file:
                        file.write(file_response.content)

            print(f"All .pha and .rsp files have been downloaded to {os.path.abspath(grb_directory)}")
            
            bat=OGIPLike("BAT",observation=os.path.join(grb_directory, pha_filename),response=os.path.join(grb_directory, rsp_filename))
            bat.set_active_measurements(grb_data['joint_active_measurement'][0])
            bat.view_count_spectrum().savefig(f'/home/polpy/polpy_test/polpy/fitting/verification/{grb_name}/{grb_name}_BAT_spectral_model.png')
            specs['BAT'] = bat

        # GBM data
        if grb_data['gbm_trigger_id'] is not None:
            
            dl = download_GBM_trigger_data(grb_data['gbm_trigger_id'], detectors=grb_data['joint_detectors'])

            for det in grb_data['joint_detectors']:
                print(f"Processing {det}...")
                polar_spec.use_effective_area_correction(grb_data['constspec_lo'], grb_data['constspec_hi'])
                polar_data.use_effective_area_correction(grb_data['constpol_lo'], grb_data['constpol_hi'])

                detectors[det] = TimeSeriesBuilder.from_gbm_tte(det, dl[det]['tte'], dl[det]['rsp'], verbose=False)
                detectors[det].set_background_interval(*grb_data['joint_bkg_intervals'])
                detectors[det].set_active_time_interval(grb_data['joint_src_interval'][0])
                detectors[det].view_lightcurve(-100, 100).savefig(f'/home/polpy/polpy_test/polpy/fitting/verification/{grb_name}/{grb_name}_lightcurve.png')

            for det in grb_data['joint_detectors']:
                specs[det] = detectors[det].to_spectrumlike()
                if det.startswith('n'):
                    specs[det].set_active_measurements(grb_data['joint_active_measurement'][0])
                else :
                    specs[det].set_active_measurements(grb_data['joint_active_measurement'][-1])

        # Just do the spectral fit first
        # modeling setup
        if grb_data['spec_prior'] == 'Band':
            specmodel = Band()

            specmodel.xp.prior = Log_normal(mu=np.log(300), sigma=np.log(100))
            specmodel.xp.bounds = (None, None)

            specmodel.K.bounds = (1E-10, None)
            specmodel.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E2)

            specmodel.alpha.bounds = (-1.5, 1.0)
            specmodel.alpha.prior = Truncated_gaussian(mu=-1, sigma=0.5, lower_bound=-1.5, upper_bound=1.0)

            specmodel.beta.bounds = (None, -1.5)
            specmodel.beta.prior = Truncated_gaussian(mu=-3., sigma=0.6, lower_bound=-5, upper_bound=-1.5)

        elif grb_data['spec_prior'] == 'CPL':
            specmodel = Cutoff_powerlaw()

            specmodel.xc.prior = Log_normal(mu=np.log(300), sigma=np.log(100))
            specmodel.xc.bounds = (None, None)

            specmodel.K.bounds = (1E-10, None)
            specmodel.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E3)

            specmodel.index.bounds = (-10, 10.0)
            specmodel.index.prior = Truncated_gaussian(mu=-1, sigma=0.5, lower_bound=-10, upper_bound=10.0)

        # pollack setup
        lp = LinearPolarization(10, 10)
        lp.angle.set_uninformative_prior(Uniform_prior)
        lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)
        lp.degree.value = 0.
        lp.angle.value = 10

        sc = SpectralComponent('synch', specmodel, lp)
        ps = PointSource('polar_GRB', 0, 0, components=[sc])

        model = Model(ps)
        if specs:
            datalist = DataList(polar_data, polar_spec, *specs.values())
        else:
            datalist = DataList(polar_data, polar_spec)

        # BAYES
        bayes = BayesianAnalysis(model, datalist)
        bayes.set_sampler("multinest")
        wrapped = [0] * len(model.free_parameters)
        wrapped[len(model.free_parameters)-1]=1

        bayes.sampler.setup(n_live_points=400,
                            resume=False,
                            importance_nested_sampling=False,
                            verbose=True,
                            wrapped_params=wrapped,
                            chain_name=f'chains/synch_p2_{grb_name}')

        bayes.sample()
        output_dir = Path('/home/polpy/polpy_test/polpy/fitting/verification/old').joinpath(grb_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        bayes.results.write_to(output_dir.joinpath(f"fit_results_{grb_name}_{i}_400_old.fits").as_posix(), overwrite=True)
        bayes.restore_median_fit()

        polar_modulation_curve_minrate20 = polar_data.display(show_model=True, min_rate=20)
        polar_modulation_curve_minrate20.savefig(output_dir.joinpath(f"{grb_name}_{i}_old_polar_modulation_curve_minrate20.png").as_posix(), bbox_inches="tight")

        bayes.results.corner_plot().savefig(output_dir.joinpath(f"{grb_name}_{i}_old_param_corner_plot.png").as_posix(), bbox_inches="tight")

        fig = display_spectrum_model_counts(bayes)
        fig.show()
        fig.savefig(output_dir.joinpath(f"{grb_name}_{i}_old_spectrum_model_counts.png").as_posix(), bbox_inches="tight")

"""

for i in range(1,2):
    with open('/home/polpy/polpy_test/polpy/fitting/GRB170114A/GRBconfig.yml', 'r') as file:
        grb_config = yaml.safe_load(file)

    import requests as reqs
    from bs4 import BeautifulSoup as bso
    import os

    # Loop through each GRB in the YAML file
    for grb_name, grb_data in grb_config.items():
        print(f"Processing {grb_name}...")

        # POLAR data and responses
        data_path = Path('/home/polpy/polpy_test/polpy/fitting').parent.joinpath('data')
        polevents = data_path.joinpath(f"POLAR_{grb_name}_NED.pevt")
        polrsp = data_path.joinpath(f"POLAR_{grb_name}_NED.prsp")
        specrsp = data_path.joinpath(f"POLAR_{grb_name}_NED.rmfarf")

        trigger_time = grb_data['trigger_time']

        # spectral
        polar = TimeSeriesBuilder.from_pol_spectrum('polar_spec', polevents.as_posix(), specrsp.as_posix(),
                                                    trigger_time=trigger_time)

        polar.set_active_time_interval(grb_data['polar_src_interval'][0])
        polar.set_background_interval(*grb_data['polar_bkg_intervals'])
        polar.create_time_bins(start=-100.0, stop=100.0, method='constant', dt=0.1)
        polar.view_lightcurve(-100, 100).savefig(f'/home/polpy/polpy_test/polpy/fitting/verification/{grb_name}/{grb_name}_lightcurve.png')

        polar_spec = polar.to_spectrumlike()
        polar_spec.set_active_measurements(grb_data['polar_active_interval'][0])
        # polarisation 
        polar_polarization_ts = TimeSeriesBuilder.from_polarization('polar_pol', polevents.as_posix(), polrsp.as_posix(),
                                                    specrsp, trigger_time=trigger_time)

        polar_polarization_ts.set_background_interval(*grb_data['polar_bkg_intervals'])
        polar_polarization_ts.set_active_time_interval(grb_data['polar_src_interval'][0])
        polar_data = polar_polarization_ts.to_polarizationlike()

        fig = polar_data.display(show_model=False, min_rate=10)
        fig.show()

        #save the figure in a folder named after the GRB
        fig.savefig(f'./{grb_name}_polarization.png')

        #moving to spectrum retrieval from co-detections

        specs={}
        detectors = {}

        # Check if 'BAT' is in joint detectors
        if 'BAT' in grb_data['joint_detectors']:

            #set to effective area correction mode
            polar_spec.use_effective_area_correction(grb_data['constspec_lo'], grb_data['constspec_hi'])
            polar_data.use_effective_area_correction(grb_data['constpol_lo'], grb_data['constpol_hi'])

            swift_trigger_number=grb_data['swift_file'][2:13]
            print(swift_trigger_number)
            base_url = f"https://swift.gsfc.nasa.gov/results/batgrbcat/GRB{grb_name}/data_product/{swift_trigger_number}-results/pha/"
            site = reqs.get(base_url)
            mysoup = bso(site.text, 'html.parser')
            pha_filename = f"sw{swift_trigger_number}b_t90.pha"
            rsp_filename = f"sw{swift_trigger_number}b_preslew.rsp"
            # Create directory for the GRB if it doesn't exist
            grb_directory = f"/home/polpy/polpy_test/polpy/data/{grb_name}/swiftBATdata"
            os.makedirs(grb_directory, exist_ok=True)

            # Get the HTML content of the page
            response = reqs.get(base_url)
            response.raise_for_status()
            soup = bso(response.text, 'html.parser')

            # Find all <a> tags with href attributes
            links = soup.find_all('a', href=True)

            # Download only files with .pha or .rsp extensions
            for link in links:
                href = link['href']
                # Filter for files with .pha or .rsp extensions
                if href.endswith(('.pha', '.rsp')):
                    file_url = base_url + href
                    file_path = os.path.join(grb_directory, href)
                    
                    print(f"Downloading {file_url}...")
                    file_response = reqs.get(file_url)
                    file_response.raise_for_status()  # Raise an error for bad responses
                    with open(file_path, 'wb') as file:
                        file.write(file_response.content)

            print(f"All .pha and .rsp files have been downloaded to {os.path.abspath(grb_directory)}")
            
            bat=OGIPLike("BAT",observation=os.path.join(grb_directory, pha_filename),response=os.path.join(grb_directory, rsp_filename))
            bat.set_active_measurements(grb_data['joint_active_measurement'][0])
            bat.view_count_spectrum().savefig(f'/home/polpy/polpy_test/polpy/fitting/verification/{grb_name}/{grb_name}_BAT_spectral_model.png')
            specs['BAT'] = bat

        # GBM data
        if grb_data['gbm_trigger_id'] is not None:
            
            dl = download_GBM_trigger_data(grb_data['gbm_trigger_id'], detectors=grb_data['joint_detectors'])

            for det in grb_data['joint_detectors']:
                print(f"Processing {det}...")
                polar_spec.use_effective_area_correction(grb_data['constspec_lo'], grb_data['constspec_hi'])
                polar_data.use_effective_area_correction(grb_data['constpol_lo'], grb_data['constpol_hi'])

                detectors[det] = TimeSeriesBuilder.from_gbm_tte(det, dl[det]['tte'], dl[det]['rsp'], verbose=False)
                detectors[det].set_background_interval(*grb_data['joint_bkg_intervals'])
                detectors[det].set_active_time_interval(grb_data['joint_src_interval'][0])
                detectors[det].view_lightcurve(-100, 100).savefig(f'/home/polpy/polpy_test/polpy/fitting/verification/{grb_name}/{grb_name}_lightcurve.png')

            for det in grb_data['joint_detectors']:
                specs[det] = detectors[det].to_spectrumlike()
                if det.startswith('n'):
                    specs[det].set_active_measurements(grb_data['joint_active_measurement'][0])
                else :
                    specs[det].set_active_measurements(grb_data['joint_active_measurement'][-1])

        # Just do the spectral fit first
        # modeling setup
        if grb_data['spec_prior'] == 'Band':
            specmodel = Band()

            specmodel.xp.prior = Log_normal(mu=np.log(300), sigma=np.log(100))
            specmodel.xp.bounds = (None, None)

            specmodel.K.bounds = (1E-10, None)
            specmodel.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E2)

            specmodel.alpha.bounds = (-1.5, 1.0)
            specmodel.alpha.prior = Truncated_gaussian(mu=-1, sigma=0.5, lower_bound=-1.5, upper_bound=1.0)

            specmodel.beta.bounds = (None, -1.5)
            specmodel.beta.prior = Truncated_gaussian(mu=-3., sigma=0.6, lower_bound=-5, upper_bound=-1.5)

        elif grb_data['spec_prior'] == 'CPL':
            specmodel = Cutoff_powerlaw()

            specmodel.xc.prior = Log_normal(mu=np.log(300), sigma=np.log(100))
            specmodel.xc.bounds = (None, None)

            specmodel.K.bounds = (1E-10, None)
            specmodel.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E3)

            specmodel.index.bounds = (-10, 10.0)
            specmodel.index.prior = Truncated_gaussian(mu=-1, sigma=0.5, lower_bound=-10, upper_bound=10.0)

        # pollack setup
        lp = LinearPolarization(10, 10)
        lp.angle.set_uninformative_prior(Uniform_prior)
        lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)
        lp.degree.value = 0.
        lp.angle.value = 10

        sc = SpectralComponent('synch', specmodel, lp)
        ps = PointSource('polar_GRB', 0, 0, components=[sc])

        model = Model(ps)
        if specs:
            datalist = DataList(polar_data, polar_spec, *specs.values())
        else:
            datalist = DataList(polar_data, polar_spec)

        # BAYES
        bayes = BayesianAnalysis(model, datalist)
        bayes.set_sampler("multinest")
        wrapped = [0] * len(model.free_parameters)
        wrapped[len(model.free_parameters)-1]=1

        bayes.sampler.setup(n_live_points=400,
                            resume=False,
                            importance_nested_sampling=False,
                            verbose=True,
                            wrapped_params=wrapped,
                            chain_name=f'chains/synch_p2_{grb_name}')

        bayes.sample()
        output_dir = Path('/home/polpy/polpy_test/polpy/fitting/verification').joinpath(grb_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        bayes.results.write_to(output_dir.joinpath(f"fit_results_{grb_name}_{i}_400.fits").as_posix(), overwrite=True)
        bayes.restore_median_fit()

        polar_modulation_curve_minrate20 = polar_data.display(show_model=True, min_rate=20)
        polar_modulation_curve_minrate20.savefig(output_dir.joinpath(f"{grb_name}_{i}_polar_modulation_curve_minrate20.png").as_posix(), bbox_inches="tight")

        bayes.results.corner_plot().savefig(output_dir.joinpath(f"{grb_name}_{i}_param_corner_plot.png").as_posix(), bbox_inches="tight")

        fig = display_spectrum_model_counts(bayes)
        fig.show()
        fig.savefig(output_dir.joinpath(f"{grb_name}_{i}_spectrum_model_counts.png").as_posix(), bbox_inches="tight")


