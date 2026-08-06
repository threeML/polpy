# PolPy

`polpy` is a universal tool to fit high-energy polarization data, based on `polarpy` ([Burgess, Kole et al. 2019, A&A](https://www.aanda.org/articles/aa/abs/2019/07/aa35056-19/aa35056-19.html)). It can be used for current and future scattering polarimeters, since it works with a standardized data and response format ([`docs/Data_and_Response_Format_For_PolPy.pdf`](docs/Data_and_Response_Format_For_PolPy.pdf)) and a defined IAU convention for the polarization angle ([`docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf`](docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf)).

`polpy` is built on [3ML](https://github.com/threeML/threeML) and currently supports POLAR and AstroSat/CZTI, but any instrument can be added, provided its data, response, and polarization angles are supplied in the standardized formats above.

## Contents

- [Installation](#installation)
- [Package layout](#package-layout)
- [Quick start](#quick-start)
- [Data and response file formats](#data-and-response-file-formats)
- [Citing](#citing)
- [License](#license)
- [Authors](#authors)

## Installation

`polpy` is available on PyPI and uses modern Python packaging via `pyproject.toml`. We recommend installing it inside a virtual environment (using `conda`, `venv`, or your preferred environment manager) to prevent dependency conflicts.

### Standard Installation

To install the latest stable version directly from PyPI, run:

```bash
pip install polpy
```

## Package layout

```
polpy/
├── polpy/
│   ├── poldata.py            # PolData: read raw polarization event files, compute PA offset
│   ├── polresponse.py        # PolResponse: read + interpolate the polarization response
│   └── polarizationlike.py   # PolarizationLike: the 3ML plugin (likelihood, fitting, display)
├── docs/
│   ├── Data_and_Response_Format_For_PolPy.pdf
│   └── Coordinate_Defintions_and_Transforms_For_PolPy.pdf
├── examples/
│   ├── example.py                 # loading and corner-plotting saved fit results
│   ├── polar_GRB170114A.py        # full joint POLAR + Fermi/GBM spectro-polarimetric fit
│   └── verify_czti_pol_fit.py     # cross-check of CZTI polarization fit against direct simulation
└── tests/                    # pytest unit tests
```

## Quick start

Condensed from `examples/polar_GRB170114A.py` — a joint POLAR polarization + POLAR/Fermi-GBM spectral fit of GRB 170114A with a Band function and a linear polarization component:

```python
from threeML import *

trigger_time = 1484431269.5000

# --- POLAR spectral time series ---
polar = TimeSeriesBuilder.from_pol_spectrum(
    'polar_spec', 'POLAR_170114A.pevt', 'POLAR_170114A.rmfarf', trigger_time=trigger_time)
polar.set_active_time_interval('-0.2-8.9')
polar.set_background_interval('-35--10', '25-75')
polar_spec = polar.to_spectrumlike()
polar_spec.set_active_measurements('30-750')
polar_spec.use_effective_area_correction(0.7, 1.3)

# --- POLAR polarization time series ---
polar_pol_ts = TimeSeriesBuilder.from_polarization(
    'polar_pol', 'POLAR_170114A.pevt', 'POLAR_170114A.rmfarf', 'POLAR_170114A.prsp',
    trigger_time=trigger_time)
polar_pol_ts.set_background_interval('-35--10', '25-75')
polar_pol_ts.set_active_time_interval('-0.2-8.9')
polar_data = polar_pol_ts.to_polarizationlike(pa_offset=0.)  # LTP -> J2000 offset, deg
polar_data.use_effective_area_correction(0.7, 1.5)

# --- model: Band spectrum + linear polarization ---
band = Band()
lp = LinearPolarization(10, 10)
lp.angle.set_uninformative_prior(Uniform_prior)
lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)

ps = PointSource('polar_GRB', 0, 0, components=[SpectralComponent('synch', band, lp)])
model = Model(ps)

datalist = DataList(polar_data, polar_spec)  # add GBM plugins here for a joint fit

bayes = BayesianAnalysis(model, datalist)
bayes.set_sampler("multinest")
bayes.sampler.setup(n_live_points=1000, resume=False)
bayes.sample()

polar_data.display(show_model=True)
```

See `examples/polar_GRB170114A.py` for the full version, including Fermi/GBM detector setup and the Band-function priors used.

## Data and response file formats

Two reference documents ship with the repo:

- [`docs/Data_and_Response_Format_For_PolPy.pdf`](docs/Data_and_Response_Format_For_PolPy.pdf) — layout of the `POLEVENTS` event FITS extension and the `.prsp` response FITS extensions (`INEBOUNDS`, `SABOUNDS`, `INPAVALS`, `SPECRESP POLMATRIX`, `SPECRESP UNPOLMATRIX`)
- [`docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf`](docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf) — instrument reference frame, local-tangent-plane frame, and the IAU polarization-angle offset convention used by `PolData.get_pa_offset()`

## Citing

If you use `polpy`, please cite the method paper it is based on:

> Burgess, J. M., Kole, M., et al. 2019, *A&A*, 627, A105 — [10.1051/0004-6361/201935056](https://www.aanda.org/articles/aa/abs/2019/07/aa35056-19/aa35056-19.html)

A dedicated instrument/software paper for `polpy` is in preparation; check back here or watch the repository for the citation once released.

## License

[GPL-3.0](LICENSE).

## Authors

Sujay Mate, Merlin Kole, Utkarsh Pathak, Yashowardhan Rai, Hancheng Li, Nicholas De Angelis, and contributors.
