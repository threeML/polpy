[![codecov](https://codecov.io/gh/threeML/polpy/graph/badge.svg)](https://codecov.io/gh/threeML/polpy)

# PolPy

`polpy` is a universal plugin for [3ML](https://github.com/threeML/threeML) developed to analyse polarization data from X-ray/gamma-ray scattering polarimeters. In particular, it is developed to perform polarimetry of high-energy transients, such as gamma-ray bursts (GRBs). It is based `polarpy` ([Burgess, Kole et al. 2019, A&A](https://www.aanda.org/articles/aa/abs/2019/07/aa35056-19/aa35056-19.html)) tool that was developed for POLAR analysis. It can be used for current and future scattering polarimeters, provided standardized data and response format ([`docs/Data_and_Response_Format_For_PolPy.pdf`](docs/Data_and_Response_Format_For_PolPy.pdf)) and a defined coordinate convention are followed ([`docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf`](docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf)).

## Contents

- [Installation](#installation)
- [Package layout](#package-layout)
- [Quick start](#quick-start)
- [Data and response file formats](#data-and-response-file-formats)
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
└── tests/                    # pytest unit tests
```

## Data and response file formats

Two reference documents ship with the repo:

- [`docs/Data_and_Response_Format_For_PolPy.pdf`](docs/Data_and_Response_Format_For_PolPy.pdf) — layout of the `POLEVENTS` event FITS extension and the `.prsp` response FITS extensions (`INEBOUNDS`, `SABOUNDS`, `INPAVALS`, `SPECRESP POLMATRIX`, `SPECRESP UNPOLMATRIX`)
- [`docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf`](docs/Coordinate_Defintions_and_Transforms_For_PolPy.pdf) — instrument reference frame, local-tangent-plane frame, and the IAU polarization-angle offset convention used by `PolData.get_pa_offset()`


## License

[GPL-3.0](LICENSE).

## Authors

Sujay Mate, Hancheng Li, Utkarsh Pathak, Yashowardhan Rai, Nicholas De Angelis, Varun Bhalerao, Merlin Kole.