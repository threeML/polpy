"""Validates PolPy's real-data joint-fit results against Figure 6 (POLAR +
Daksha 5-face joint polarization fit, injection-recovery study for
GRB160325A) of the polpy paper.

Model setup, instrument toggles, and file path conventions mirror
fitting/GRB160325A/joint_fit_daksha_polar.py from
https://github.com/dakshasat/polpy_joint_fit_paper (private repo; the
underlying POLAR/Daksha data files live only on the team's analysis server
and are not part of this checkout). These tests skip themselves if the data
isn't present at those paths -- they're meant to be run on a machine that
has both the real data and a working MultiNest install, not in ordinary CI.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from astromodels import Model, PointSource
from astromodels.core.polarization import LinearPolarization
from astromodels.core.spectral_component import SpectralComponent
from astromodels.functions.functions_1D.powerlaws import Band
from threeML import BayesianAnalysis, DataList, Truncated_gaussian, Uniform_prior
from threeML.utils.data_builders import TimeSeriesBuilder

from polpy.poldata import PolData

# =========================================================
# Paths & config, mirroring joint_fit_daksha_polar.py exactly
# =========================================================
GRB_NAME = "GRB160325A"
DAKSHA_FACES_5 = [11, 4, 12, 0, 1]

MULT = 10.0

PA_LOCAL = "104"  # sky PA 90, matching Figure 6's injected PA=90
SKY_PA = 90

DAKSHA_DATA_DIR = "/home/polpy/daksha_data"
POLAR_DATA_DIR = "/home/polpy/polar_data/160325A/new_data_20260717/new_data_20260717"
POLAR_RESPONSE = os.path.join(POLAR_DATA_DIR, "POLAR_160325A.prsp")

POLAR_TRIGGER_TIME = 1474786761.0
DAKSHA_TRIGGER_TIME = 0.0

BASE_K = 0.017
XP_VAL = 266.04
ALPHA_VAL = -0.77
BETA_VAL = -2.67

# =========================================================
# Figure 6 quoted results: median^{+plus}_{-minus}, 10x fluence, sky PA=90
# =========================================================
QUOTED_RESULTS = {
    (0.8, "POLAR_only"): {"pf": (87.0, 6.0, 6.3), "pa": (90.4, 1.0, 0.9)},
    (0.8, "DAKSHA_5f_only"): {"pf": (77.1, 6.0, 6.0), "pa": (91.5, 2.0, 2.1)},
    (0.8, "JOINT"): {"pf": (81.8, 4.6, 4.7), "pa": (91.1, 1.0, 1.0)},
    (0.5, "POLAR_only"): {"pf": (42.7, 7.0, 7.5), "pa": (89.4, 2.2, 2.3)},
    (0.5, "DAKSHA_5f_only"): {"pf": (44.1, 6.3, 5.9), "pa": (90.6, 3.7, 3.5)},
    (0.5, "JOINT"): {"pf": (44.2, 4.8, 4.7), "pa": (91.0, 1.9, 1.8)},
}

# Relative tolerance for comparing the *width* of the computed error bars
# against the paper's quoted error bars. MCMC posteriors are stochastic
# (different multinest seed/live-point count run-to-run), so this checks
# "same order of uncertainty", not a bit-exact match.
ERROR_WIDTH_RTOL = 0.4


def _polar_pevt_path(pf, pa_local):
    n_str = f"N{int(MULT):03d}"
    p_str = f"P{int(round(pf * 100)):03d}"
    a_str = f"A{SKY_PA:03d}"
    return os.path.join(POLAR_DATA_DIR, f"POLAR_160325A-{n_str}-{p_str}-{a_str}.pevt")


def _daksha_paths(face, pf, pa_local):
    pevt = os.path.join(
        DAKSHA_DATA_DIR,
        f"{GRB_NAME}_PF_{pf}_{MULT:.1f}x",
        f"{GRB_NAME}_PA_{pa_local}_PF_{pf}_face_{face}_{MULT:.1f}x_dakshapol.pevt",
    )
    prsp = os.path.join(
        DAKSHA_DATA_DIR,
        f"{GRB_NAME}_response",
        f"face_{face}",
        f"DAKSHA_POLRSP_EMIN_100_EMAX_1000_{GRB_NAME}_{face}.prsp",
    )
    return pevt, prsp


def _required_paths(use_polar, use_daksha, pf, pa_local):
    paths = []
    if use_polar:
        paths += [_polar_pevt_path(pf, pa_local), POLAR_RESPONSE]
    if use_daksha:
        for face in DAKSHA_FACES_5:
            paths += list(_daksha_paths(face, pf, pa_local))
    return paths


def _prepare_datalist(use_polar, use_daksha, pf, pa_local):
    datalist_items = []

    if use_polar:
        polar_pevt = _polar_pevt_path(pf, pa_local)
        polar_ts = TimeSeriesBuilder.from_polarization(
            name="polar_pol",
            polevents=polar_pevt,
            polrsp=POLAR_RESPONSE,
            trigger_time=POLAR_TRIGGER_TIME,
        )
        polar_ts.set_background_interval("-300.0--0.0", "50-300")
        polar_ts.set_active_time_interval("0.5-40.0")
        datalist_items.append(polar_ts.to_polarizationlike())

    if use_daksha:
        for face in DAKSHA_FACES_5:
            daksha_pevt, daksha_prsp = _daksha_paths(face, pf, pa_local)
            daksha_ts = TimeSeriesBuilder.from_polarization(
                name=f"daksha_pol_face{face}",
                polevents=daksha_pevt,
                polrsp=daksha_prsp,
                trigger_time=DAKSHA_TRIGGER_TIME,
            )
            daksha_ts.set_active_time_interval("-20.0 - 20.0")
            daksha_ts.set_background_interval("-245 - -90")
            datalist_items.append(daksha_ts.to_polarizationlike())

    return DataList(*datalist_items)


def _build_model(pf, sky_pa):
    band = Band()
    band.xp.bounds = (10.0, 1000.0)
    band.xp.prior = Truncated_gaussian(
        mu=XP_VAL, sigma=XP_VAL * 0.05, lower_bound=XP_VAL * 0.95, upper_bound=XP_VAL * 1.05
    )
    band.xp.value = XP_VAL

    band.K.bounds = (1e-5, 10.0)
    band.K.prior = Truncated_gaussian(
        mu=BASE_K * MULT,
        sigma=BASE_K * MULT * 0.05,
        lower_bound=BASE_K * MULT * 0.95,
        upper_bound=BASE_K * MULT * 1.05,
    )
    band.K.value = BASE_K * MULT

    band.alpha.bounds = (-2.5, 1.0)
    band.alpha.prior = Truncated_gaussian(
        mu=ALPHA_VAL,
        sigma=abs(ALPHA_VAL) * 0.05,
        lower_bound=min(ALPHA_VAL * 0.95, ALPHA_VAL * 1.05),
        upper_bound=max(ALPHA_VAL * 0.95, ALPHA_VAL * 1.05),
    )
    band.alpha.value = ALPHA_VAL

    band.beta.bounds = (-5.0, -1.5)
    band.beta.prior = Truncated_gaussian(
        mu=BETA_VAL,
        sigma=abs(BETA_VAL) * 0.05,
        lower_bound=min(BETA_VAL * 0.95, BETA_VAL * 1.05),
        upper_bound=max(BETA_VAL * 0.95, BETA_VAL * 1.05),
    )
    band.beta.value = BETA_VAL

    lp = LinearPolarization(pf * 100.0, sky_pa)
    lp.angle.prior = Uniform_prior(lower_bound=0.0, upper_bound=180.0)
    lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)

    sc = SpectralComponent("synch", band, lp)
    ps = PointSource(GRB_NAME, 0, 0, components=[sc])
    return Model(ps)


def _quantiles(samples):
    q16, q50, q84 = np.percentile(samples, [16, 50, 84])
    return q50, q84 - q50, q50 - q16


@pytest.mark.slow
@pytest.mark.parametrize(
    "pf,mode",
    [
        (0.8, "POLAR_only"),
        (0.8, "DAKSHA_5f_only"),
        (0.8, "JOINT"),
        (0.5, "POLAR_only"),
        (0.5, "DAKSHA_5f_only"),
        (0.5, "JOINT"),
    ],
)
def test_paper_figure6_recovery(pf, mode):
    """For each instrument mode / injected PF, run the real Bayesian
    polarization fit and check the recovered PF/PA against Figure 6:
    the posterior median must be within 1-sigma (using the paper's quoted
    uncertainty) of the quoted value, and the computed error bar width must
    be the same order of magnitude as the quoted one (both + and - side)."""
    use_polar = mode in ("POLAR_only", "JOINT")
    use_daksha = mode in ("DAKSHA_5f_only", "JOINT")

    required = _required_paths(use_polar, use_daksha, pf, PA_LOCAL)
    if not all(os.path.exists(p) for p in required):
        pytest.skip(
            f"Real POLAR/Daksha data for {GRB_NAME} (PF={pf}, mode={mode}) not "
            "found locally. This test validates against real analysis-server "
            "data only; see module docstring."
        )

    datalist = _prepare_datalist(use_polar, use_daksha, pf, PA_LOCAL)
    model = _build_model(pf, SKY_PA)

    bayes = BayesianAnalysis(model, datalist)
    bayes.set_sampler("multinest")
    wrapped = [0] * len(model.free_parameters)
    bayes.sampler.setup(
        n_live_points=1000,
        resume=False,
        importance_nested_sampling=False,
        verbose=False,
        wrapped_params=wrapped,
    )
    bayes.sample()

    df = bayes.results.get_data_frame()
    degree_col = [c for c in df.columns if "degree" in c.lower()][0]
    angle_col = [c for c in df.columns if "angle" in c.lower()][0]
    pf_samples = df[degree_col].values * 100.0  # fraction -> percent, if needed
    pa_samples = df[angle_col].values

    pf_median, pf_plus, pf_minus = _quantiles(pf_samples)
    pa_median, pa_plus, pa_minus = _quantiles(pa_samples)

    q = QUOTED_RESULTS[(pf, mode)]
    q_pf_median, q_pf_plus, q_pf_minus = q["pf"]
    q_pa_median, q_pa_plus, q_pa_minus = q["pa"]

    # Median within 1-sigma (quoted uncertainty) of the quoted value
    assert abs(pf_median - q_pf_median) <= max(q_pf_plus, q_pf_minus), (
        f"PF median {pf_median:.1f} not within 1-sigma of quoted {q_pf_median}"
        f" (+{q_pf_plus}/-{q_pf_minus})"
    )
    assert abs(pa_median - q_pa_median) <= max(q_pa_plus, q_pa_minus), (
        f"PA median {pa_median:.1f} not within 1-sigma of quoted {q_pa_median}"
        f" (+{q_pa_plus}/-{q_pa_minus})"
    )

    # Same average error range (both + and - side) as quoted, within tolerance
    computed_pf_avg_err = (pf_plus + pf_minus) / 2.0
    quoted_pf_avg_err = (q_pf_plus + q_pf_minus) / 2.0
    assert computed_pf_avg_err == pytest.approx(quoted_pf_avg_err, rel=ERROR_WIDTH_RTOL)

    computed_pa_avg_err = (pa_plus + pa_minus) / 2.0
    quoted_pa_avg_err = (q_pa_plus + q_pa_minus) / 2.0
    assert computed_pa_avg_err == pytest.approx(quoted_pa_avg_err, rel=ERROR_WIDTH_RTOL)


# =========================================================
# PA offset regression pins
# =========================================================
# get_pa_offset() computed by PolData must match these values (to a tight
# tolerance -- rel=1e-9, loose enough to absorb last-ULP floating point noise
# between platforms/math libraries, tight enough to catch any real
# regression) for future code versions. Any change to PolData's
# coordinate-transform logic that shifts them is a breaking change and must
# be caught here.

# --- tests/data/POLAR_160325A.pevt (see tests/test_real_data_joint_fit.py) ---
# Computed with the current version of PolData against this file (same
# instrument-pointing geometry as the GRB160325A_PA131_PF0.5 POLAR file
# above -- get_pa_offset() depends only on RAX/DECX/RAZ/DECZ/RAGRB/DECGRB,
# not on the injected PF/PA event data, so the two values matching is
# expected, not a coincidence).
CURRENT_PA_OFFSET_POLAR_160325A = 42.73256796848862


def test_pa_offset_regression_polar_160325a():
    """PolData.get_pa_offset() must match (see tolerance note above) the
    value computed by the current version of the code for this real POLAR
    event file."""
    data_path = Path(__file__).parent / "data" / "POLAR_160325A.pevt"
    if not data_path.exists():
        pytest.skip("tests/data/POLAR_160325A.pevt not found.")

    data = PolData(str(data_path))
    pa_offset = data.get_pa_offset()

    assert pa_offset == pytest.approx(CURRENT_PA_OFFSET_POLAR_160325A, rel=1e-9)


# --- tests/data/GRB160325A_PA131_PF0.5/ (POLAR + 5 Daksha faces) ---
# Computed directly with the current version of PolData against the real
# data provided on 2026-08-17 (10x fluence, injected local PA=131 / sky
# PA=117, PF=0.5). Since these were computed by us rather than supplied as
# an external reference, they pin *current* behavior -- a future change that
# shifts them still needs deliberate review, same as the POLAR_160325A pin.
JOINT_DATA_DIR = Path(__file__).parent / "data" / "GRB160325A_PA131_PF0.5"

PA_OFFSET_REGRESSION_VALUES = {
    "polar": ("POLAR_160325A-N010-P050-A117.pevt", 42.73256796848862),
    "daksha_face0": ("GRB160325A_PA_131_PF_0.5_face_0_10.0x_dakshapol.pevt", 14.395386074115402),
    "daksha_face1": ("GRB160325A_PA_131_PF_0.5_face_1_10.0x_dakshapol.pevt", -32.348659599397834),
    "daksha_face4": ("GRB160325A_PA_131_PF_0.5_face_4_10.0x_dakshapol.pevt", 20.524149829946854),
    "daksha_face11": ("GRB160325A_PA_131_PF_0.5_face_11_10.0x_dakshapol.pevt", 179.98621327451008),
    "daksha_face12": ("GRB160325A_PA_131_PF_0.5_face_12_10.0x_dakshapol.pevt", -93.76014956293814),
}


@pytest.mark.parametrize(
    "filename,expected",
    [v for v in PA_OFFSET_REGRESSION_VALUES.values()],
    ids=list(PA_OFFSET_REGRESSION_VALUES.keys()),
)
def test_pa_offset_regression_grb160325a_joint(filename, expected):
    """PolData.get_pa_offset() must exactly match the value computed by the
    current production version of the code for each real POLAR/Daksha event
    file in the GRB160325A joint (local PA=131, PF=0.5, 10x) dataset."""
    data_path = JOINT_DATA_DIR / filename
    if not data_path.exists():
        pytest.skip(f"{filename} not found in tests/data/GRB160325A_PA131_PF0.5/.")

    data = PolData(str(data_path))
    assert data.get_pa_offset() == pytest.approx(expected, rel=1e-9)


# =========================================================
# Real local MCMC verification (POLAR + 5-face Daksha, real data)
# =========================================================
def _real_multinest_available():
    """True only if pymultinest can actually load a real libmultinest (not
    the inert stub conftest.py substitutes in when the native library is
    missing, e.g. on Windows -- see conftest.py's _ensure_importable_pymultinest)."""
    try:
        import pymultinest

        return pymultinest.run is not None
    except Exception:
        return False


@pytest.mark.slow
def test_grb160325a_joint_mcmc_recovery():
    """Runs the real joint POLAR + 5-face Daksha MultiNest fit on the
    GRB160325A_PA131_PF0.5 dataset (local PA=131, sky PA=117, PF=0.5, 10x
    fluence) and checks the recovered PF/PA are within 1-sigma of the known
    injected truth. Requires a real MultiNest install; skips otherwise."""
    if not _real_multinest_available():
        pytest.skip("Real MultiNest library not available in this environment.")

    polar_pevt = JOINT_DATA_DIR / "POLAR_160325A-N010-P050-A117.pevt"
    polar_prsp = JOINT_DATA_DIR / "POLAR_160325A.prsp"
    if not polar_pevt.exists() or not polar_prsp.exists():
        pytest.skip("GRB160325A_PA131_PF0.5 data not found in tests/data/.")

    injected_pf, injected_pa = 50.0, 117.0
    faces = [11, 4, 12, 0, 1]

    datalist_items = []
    polar_ts = TimeSeriesBuilder.from_polarization(
        name="polar_pol",
        polevents=str(polar_pevt),
        polrsp=str(polar_prsp),
        trigger_time=POLAR_TRIGGER_TIME,
    )
    polar_ts.set_background_interval("-300.0--0.0", "50-300")
    polar_ts.set_active_time_interval("0.5-40.0")
    datalist_items.append(polar_ts.to_polarizationlike())

    for face in faces:
        daksha_pevt = JOINT_DATA_DIR / f"GRB160325A_PA_131_PF_0.5_face_{face}_10.0x_dakshapol.pevt"
        daksha_prsp = JOINT_DATA_DIR / f"DAKSHA_POLRSP_EMIN_100_EMAX_1000_GRB160325A_{face}.prsp"
        daksha_ts = TimeSeriesBuilder.from_polarization(
            name=f"daksha_pol_face{face}",
            polevents=str(daksha_pevt),
            polrsp=str(daksha_prsp),
            trigger_time=DAKSHA_TRIGGER_TIME,
        )
        daksha_ts.set_active_time_interval("-20.0 - 20.0")
        daksha_ts.set_background_interval("-245 - -90")
        datalist_items.append(daksha_ts.to_polarizationlike())

    datalist = DataList(*datalist_items)
    model = _build_model(0.5, 117)

    bayes = BayesianAnalysis(model, datalist)
    bayes.set_sampler("multinest")
    wrapped = [0] * len(model.free_parameters)
    bayes.sampler.setup(
        n_live_points=400,
        resume=False,
        importance_nested_sampling=False,
        verbose=False,
        wrapped_params=wrapped,
    )
    bayes.sample()

    df = bayes.results.get_data_frame()
    degree_row = [i for i in df.index if "degree" in i.lower()][0]
    angle_row = [i for i in df.index if "angle" in i.lower()][0]

    pf_val = df.loc[degree_row, "value"]
    pf_plus = df.loc[degree_row, "positive_error"]
    pf_minus = abs(df.loc[degree_row, "negative_error"])

    pa_val = df.loc[angle_row, "value"]
    pa_plus = df.loc[angle_row, "positive_error"]
    pa_minus = abs(df.loc[angle_row, "negative_error"])

    assert abs(pf_val - injected_pf) <= max(pf_plus, pf_minus), (
        f"PF median {pf_val:.1f} not within 1-sigma of injected {injected_pf}"
    )
    assert abs(pa_val - injected_pa) <= max(pa_plus, pa_minus), (
        f"PA median {pa_val:.1f} not within 1-sigma of injected {injected_pa}"
    )
