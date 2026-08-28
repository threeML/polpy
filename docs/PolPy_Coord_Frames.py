#!/usr/bin/env python3
"""
PolPy_Coord_Frames_updated.pdf
==============================

Two-panel schematic of the reference frames used in polpy, drawn with a
consistent 3-D geometry and a custom orthographic projection.

Left panel : celestial (RA-Dec / J2000) frame [brown] and the Instrument
             Reference Frame (IRF) [pink], both centred on the sphere origin
             (satellite-centric); the source S [dashed red] at (alpha, delta);
             the IAU tangent-plane frame [green] at the source, with
             N_IAU pointing towards +Z_J2000 and the frame z-axis along +S.

Right panel: the same sphere drawn in the IRF [pink]; the source S at
             (theta, phi) -- theta measured from the +Z_IRF pole; the LTP
             tangent-plane frame [blue] with N_LTP pointing towards +Z_IRF
             and D_LTP = -S (NED, right-handed); the IAU frame [green] is
             repeated to show the offset psi_0 between the two norths.

All tangent-frame vectors are computed from a single rotation matrix R that
relates the IRF to the J2000 frame, so handedness, angles and the psi_0
offset are geometrically exact (angles chosen for clarity, not for any
particular observation).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "dejavusans"
rcParams["font.family"] = "sans-serif"
rcParams["pdf.fonttype"] = 42

# ----------------------------------------------------------------------
# 1. Scene parameters (all angles in degrees)
# ----------------------------------------------------------------------
ALPHA, DELTA = 75.0, 42.0  # source RA, Dec in J2000 (delta from equator)
THETA, PHI = 60.0, 65.0  # source theta, phi in IRF (theta from +Z pole)
CHI = 30.0  # free spin of the IRF about the source direction
PSI_IAU = 28.0  # polarization angle, measured N_IAU -> E_IAU

CAM_AZ, CAM_EL = 25.0, 8.0  # orthographic camera direction (both panels)

# ---- colours ---------------------------------------------------------
C_J2000 = "#8B4513"  # brown  : RA-Dec (J2000) frame
C_IRF = "#E020B0"  # pink   : Instrument Reference Frame
C_IAU = "#1E8B22"  # green  : IAU tangent frame
C_LTP = "#1230CC"  # blue   : LTP tangent frame
C_SRC = "#E01010"  # red    : source vector
C_POL = "#F5A11A"  # orange : polarization direction
C_RAFILL = "#C9B8D8"  # lavender fill for alpha / phi wedges
C_RAARC = "#7B2D8B"  # purple arc arrow for alpha / phi
C_DEFILL = "#ECFBA8"  # peach fill for delta / theta wedges
C_DEARC = "#7C9700"  # orange arc arrow for delta / theta
C_SPHERE = "black"

WHITE_STROKE = [pe.withStroke(linewidth=2.8, foreground="white")]


# ----------------------------------------------------------------------
# 2. Basic geometry helpers
# ----------------------------------------------------------------------
def sph(az_deg, el_deg):
    """Unit vector from azimuth (from +X, in XY plane) and elevation."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def rodrigues(axis, ang_rad):
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(ang_rad) * K + (1 - np.cos(ang_rad)) * (K @ K)


def tangent_frame(r, zref):
    """North/East unit vectors of the tangent plane at r, w.r.t. pole zref.

    e points towards increasing azimuth ('east'), n towards the zref pole.
    Note n x e = -r for both frames: in the tangent plane the IAU and LTP
    frames are identically oriented; they differ in the choice of third
    axis (+S for IAU -> left-handed triple, D = -S for LTP -> right-handed
    NED triple).
    """
    e = np.cross(zref, r)
    e = e / np.linalg.norm(e)
    n = np.cross(r, e)
    return n, e


# ---- source & IRF orientation ---------------------------------------
ZHAT = np.array([0.0, 0.0, 1.0])
r_J = sph(ALPHA, DELTA)  # source direction, J2000 components
r_I = sph(PHI, 90.0 - THETA)  # source direction, IRF components

# rotation R (J2000 <- IRF):  R @ v_IRF = v_J2000,  with  R @ r_I = r_J
axis0 = np.cross(r_I, r_J)
ang0 = np.arccos(np.clip(r_I @ r_J, -1.0, 1.0))
R0 = rodrigues(axis0, ang0)
R = R0 @ rodrigues(r_I, np.radians(CHI))

# IRF axes in J2000 components (left panel)
xI_J, yI_J, zI_J = R[:, 0], R[:, 1], R[:, 2]
# J2000 pole in IRF components (right panel)
zJ_I = R.T @ ZHAT

# tangent frames
nIAU_J, eIAU_J = tangent_frame(r_J, ZHAT)  # left panel (J2000 world)
nLTP_I, eLTP_I = tangent_frame(r_I, ZHAT)  # right panel (IRF world)
nIAU_I, eIAU_I = tangent_frame(r_I, zJ_I)  # IAU frame in IRF components

# polarization direction (fixed physical direction in the tangent plane)
p_J = np.cos(np.radians(PSI_IAU)) * nIAU_J + np.sin(np.radians(PSI_IAU)) * eIAU_J
p_I = R.T @ p_J

PSI_LTP = np.degrees(np.arctan2(p_I @ eLTP_I, p_I @ nLTP_I))
PSI_0 = np.degrees(np.arctan2(nIAU_I @ eLTP_I, nIAU_I @ nLTP_I))
print(
    f"psi_IAU = {PSI_IAU:.1f} deg, psi_0 = {PSI_0:.1f} deg, "
    f"psi_LTP = {PSI_LTP:.1f} deg (= psi_IAU + psi_0)"
)


# ----------------------------------------------------------------------
# 3. Orthographic projection
# ----------------------------------------------------------------------
class View:
    def __init__(self, az, el):
        self.c = sph(az, el)  # towards camera
        sx = np.cross(ZHAT, self.c)
        self.sx = sx / np.linalg.norm(sx)  # screen right
        self.sy = np.cross(self.c, self.sx)  # screen up

    def P(self, p):
        p = np.atleast_2d(p)
        return np.column_stack([p @ self.sx, p @ self.sy])

    def depth(self, p):
        return np.atleast_2d(p) @ self.c


VIEW = View(CAM_AZ, CAM_EL)


# ----------------------------------------------------------------------
# 4. Drawing helpers
# ----------------------------------------------------------------------
def split_curve(ax, pts3, near_kw, far_kw):
    """Plot a 3-D polyline, near side with near_kw, hidden side with far_kw."""
    uv = VIEW.P(pts3)
    d = VIEW.depth(pts3).ravel()
    near = d >= 0

    for mask, kw in ((near, near_kw), (~near, far_kw)):
        if kw is None:
            continue
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # contiguous runs
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in runs:
            ax.plot(uv[run, 0], uv[run, 1], **kw)


def great_circle(ax, u, w, near_kw, far_kw):
    t = np.linspace(0, 2 * np.pi, 361)
    pts = np.outer(np.cos(t), u) + np.outer(np.sin(t), w)
    split_curve(ax, pts, near_kw, far_kw)


def arc3(u, w, t0_deg, t1_deg, n=120, radius=1.0):
    t = np.radians(np.linspace(t0_deg, t1_deg, n))
    return radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), w))


def fan(ax, u, w, t0_deg, t1_deg, fc, alpha):
    pts = np.vstack([[0, 0, 0], arc3(u, w, t0_deg, t1_deg)])
    uv = VIEW.P(pts)
    ax.fill(uv[:, 0], uv[:, 1], fc=fc, alpha=alpha, ec="none", zorder=1)


def arc_arrow(
    ax, u, w, t0_deg, t1_deg, color, lw=1.5, radius=1.0, mutation=13, zorder=6
):
    pts = arc3(u, w, t0_deg, t1_deg, radius=radius)
    uv = VIEW.P(pts)
    ax.plot(
        uv[:-4, 0],
        uv[:-4, 1],
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.annotate(
        "",
        xy=uv[-1],
        xytext=uv[-6],
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=mutation,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )


def arrow3(
    ax, p0, p1, color, lw=1.5, ls="-", mutation=17, zorder=5, double_sided=False
):
    uv0, uv1 = VIEW.P(p0)[0], VIEW.P(p1)[0]
    if double_sided:
        ax.annotate(
            "",
            xy=uv1,
            xytext=uv0,
            arrowprops=dict(
                arrowstyle="<|-|>",
                color=color,
                lw=lw,
                linestyle=ls,
                mutation_scale=mutation,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=zorder,
        )

    else:
        ax.annotate(
            "",
            xy=uv1,
            xytext=uv0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                linestyle=ls,
                mutation_scale=mutation,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=zorder,
        )


def label3(
    ax,
    p3,
    text,
    color,
    dx=0.0,
    dy=0.0,
    fs=13,
    ha="center",
    va="center",
    zorder=10,
    weight="normal",
):
    uv = VIEW.P(p3)[0]
    ax.text(
        uv[0] + dx,
        uv[1] + dy,
        text,
        color=color,
        fontsize=fs,
        ha=ha,
        va=va,
        zorder=zorder,
        path_effects=WHITE_STROKE,
        fontweight=weight,
    )


def label_along(ax, u, w, t_deg, text, fs=10.5, color="0.25", zorder=4, offset=0.05):
    """Italic label placed on a great circle, rotated along the local tangent."""
    p = arc3(u, w, t_deg - 4, t_deg + 4, n=3)
    uv = VIEW.P(p)
    dxy = uv[2] - uv[0]
    ang = np.degrees(np.arctan2(dxy[1], dxy[0]))
    if ang > 90 or ang < -90:
        ang += 180
    nrm = np.array([-dxy[1], dxy[0]])
    nrm = nrm / np.linalg.norm(nrm) * offset
    ax.text(
        uv[1][0] + nrm[0],
        uv[1][1] + nrm[1],
        text,
        fontsize=fs,
        color=color,
        style="italic",
        rotation=ang,
        ha="center",
        va="center",
        rotation_mode="anchor",
        zorder=zorder,
        path_effects=WHITE_STROKE,
    )


def psi_arc(
    ax, P, d0, d1, radius, color, text, t_frac=0.55, lab_r=1.45, fs=12, zorder=8
):
    """Small angle arc at point P from tangent direction d0 to d1."""
    ang = np.arctan2(np.linalg.norm(np.cross(d0, d1)), d0 @ d1)
    w = d1 - (d1 @ d0) * d0
    w = w / np.linalg.norm(w)
    t = np.linspace(0, ang, 40)
    pts = P + radius * (np.outer(np.cos(t), d0) + np.outer(np.sin(t), w))
    uv = VIEW.P(pts)
    ax.plot(uv[:-3, 0], uv[:-3, 1], color=color, lw=1.4, zorder=zorder)
    ax.annotate(
        "",
        xy=uv[-1],
        xytext=uv[-4],
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.3,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )
    mid = P + lab_r * radius * (np.cos(t_frac * ang) * d0 + np.sin(t_frac * ang) * w)
    label3(ax, mid, text, color, fs=fs, zorder=zorder + 2)


def sphere_base(ax):
    """Sphere outline + origin dot."""
    circ = plt.Circle((0, 0), 1.0, fill=False, color=C_SPHERE, lw=1.1, zorder=3)
    ax.add_patch(circ)
    ax.plot(0, 0, "o", color="black", ms=3.5, zorder=6)


NEAR = dict(color=C_SPHERE, lw=0.9, zorder=3)
FAR = dict(color="0.55", lw=0.7, ls=(0, (2, 3)), zorder=2)

# ======================================================================
# 5. Build the figure
# ======================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 7.0))
for ax in (axL, axR):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.52, 1.60)
    ax.set_ylim(-1.45, 1.55)

XHAT, YHAT = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
AXLEN = 1.30  # length of the frame axes (sphere radius = 1)
TLEN = 0.36  # length of tangent-frame vectors
SRCLEN = 1.6  # length of the source arrow

# ----------------------------------------------------------------------
# LEFT PANEL : J2000 world.  RA-Dec frame + IRF + S + IAU frame.
# ----------------------------------------------------------------------
ax = axL

# shaded wedges for alpha (equatorial plane) and delta (source meridian)
foot_J = sph(ALPHA, 0.0)  # equatorial foot of the source
fan(ax, XHAT, YHAT, 0, ALPHA, C_RAFILL, 0.45)
fan(ax, foot_J, ZHAT, 0, DELTA, C_DEFILL, 0.50)

# great circles: equator, prime meridian, source meridian
great_circle(ax, XHAT, YHAT, NEAR, FAR)  # celestial equator
great_circle(ax, XHAT, ZHAT, NEAR, FAR)  # prime meridian
great_circle(
    ax,
    foot_J,
    ZHAT,
    dict(color="0.35", lw=0.6, zorder=3),
    dict(color="0.7", lw=0.5, ls=(0, (2, 3)), zorder=2),
)
sphere_base(ax)

label_along(ax, XHAT, YHAT, -35, "Celestial Equator", offset=0.07)
label_along(ax, XHAT, ZHAT, 35, "Prime Meridian", offset=0.07)

# arc arrows + angle labels
arc_arrow(ax, XHAT, YHAT, 3, ALPHA - 2, C_RAARC)
arc_arrow(ax, foot_J, ZHAT, 2, DELTA - 11, C_DEARC)
label3(ax, 0.60 * sph(ALPHA / 2, 0), r"$\alpha$", C_RAARC, dy=-0.05, fs=15)
label3(ax, 1.10 * sph(ALPHA, DELTA / 2), r"$\delta$", C_DEARC, fs=15)

# J2000 axes (brown)
arrow3(ax, [0, 0, 0], AXLEN * XHAT, C_J2000)
arrow3(ax, [0, 0, 0], AXLEN * YHAT, C_J2000)
arrow3(ax, [0, 0, 0], AXLEN * ZHAT, C_J2000)
label3(ax, AXLEN * XHAT, r"$\mathrm{X_{J2000}}$", C_J2000, dx=-0.02, dy=-0.10)
label3(ax, AXLEN * YHAT, r"$\mathrm{Y_{J2000}}$", C_J2000, dx=0.13, dy=-0.05)
label3(ax, AXLEN * ZHAT, r"$\mathrm{Z_{J2000}}$", C_J2000, dy=0.09)

# IRF axes (pink) -- centred at the sphere origin (satellite-centric!)
arrow3(ax, [0, 0, 0], AXLEN * xI_J, C_IRF)
arrow3(ax, [0, 0, 0], AXLEN * yI_J, C_IRF)
arrow3(ax, [0, 0, 0], AXLEN * zI_J, C_IRF)
label3(ax, AXLEN * xI_J, r"$\mathrm{X_{IRF}}$", C_IRF, dx=0.02, dy=-0.10)
label3(ax, AXLEN * yI_J, r"$\mathrm{Y_{IRF}}$", C_IRF, dx=0.13, dy=-0.04)
label3(ax, AXLEN * zI_J, r"$\mathrm{Z_{IRF}}$", C_IRF, dy=0.09)

# source vector (dashed red), + a thin dashed projection onto the equator
uvO, uvF = VIEW.P([0, 0, 0])[0], VIEW.P(0.99 * foot_J)[0]
ax.plot(
    [uvO[0], uvF[0]], [uvO[1], uvF[1]], color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=4
)
arrow3(ax, [0, 0, 0], SRCLEN * r_J, C_SRC, ls="--", lw=1.5)
ax.plot(*VIEW.P(r_J)[0], "o", color="black", ms=3, zorder=9)
label3(ax, SRCLEN * r_J, r"$S\,(\alpha,\delta)$", C_SRC, dx=0.12, dy=0.05, fs=13.5)

# IAU tangent frame at the source (green); its z-axis is +S itself
P = r_J
arrow3(ax, P, P + (TLEN + 0.1) * nIAU_J, C_IAU, lw=1.5, zorder=7)
arrow3(ax, P, P + (TLEN + 0.1) * eIAU_J, C_IAU, lw=1.5, zorder=7)
label3(ax, P + (TLEN + 0.1) * nIAU_J, r"$\mathrm{N_{IAU}}$", C_IAU, dx=-0.1, dy=0.01)
label3(ax, P + (TLEN + 0.1) * eIAU_J, r"$\mathrm{E_{IAU}}$", C_IAU, dx=0.11, dy=0.05)

# polarization direction (orange, dashed, double-headed) + psi_IAU
arrow3(
    ax,
    P - 0.2 * p_J,
    P + 0.35 * p_J,
    C_POL,
    ls="-",
    lw=1.6,
    mutation=12,
    zorder=8,
    double_sided=True,
)

psi_arc(
    ax, P, nIAU_J, p_J, 0.27, C_IAU, r"$\psi_{\mathrm{IAU}}$", lab_r=1.45, t_frac=1.10
)

# ----------------------------------------------------------------------
# RIGHT PANEL : IRF world.  IRF + S + LTP frame + IAU frame.
# ----------------------------------------------------------------------
ax = axR

foot_I = sph(PHI, 0.0)
fan(ax, XHAT, YHAT, 0, PHI, C_RAFILL, 0.45)  # phi wedge
fan(ax, ZHAT, foot_I, 0, THETA, C_DEFILL, 0.50)  # theta wedge (from pole!)

great_circle(ax, XHAT, YHAT, NEAR, FAR)  # IRF equator
great_circle(ax, XHAT, ZHAT, NEAR, FAR)  # IRF prime meridian
great_circle(
    ax,
    foot_I,
    ZHAT,
    dict(color="0.35", lw=0.6, zorder=3),
    dict(color="0.7", lw=0.5, ls=(0, (2, 3)), zorder=2),
)
sphere_base(ax)

arc_arrow(ax, XHAT, YHAT, 3, PHI - 2, C_RAARC)
arc_arrow(ax, ZHAT, foot_I, 4, 20, C_DEARC, radius=0.95)  # from the pole
label3(ax, 0.60 * sph(PHI / 2, 0), r"$\phi$", C_RAARC, dy=-0.05, fs=15)
label3(ax, 0.85 * sph(PHI, 90 - 10), r"$\theta$", C_DEARC, fs=15)

# IRF axes (pink), canonical orientation in this panel
arrow3(ax, [0, 0, 0], AXLEN * XHAT, C_IRF)
arrow3(ax, [0, 0, 0], AXLEN * YHAT, C_IRF)
arrow3(ax, [0, 0, 0], AXLEN * ZHAT, C_IRF)
label3(ax, AXLEN * XHAT, r"$\mathrm{X_{IRF}}$", C_IRF, dx=-0.02, dy=-0.10)
label3(ax, AXLEN * YHAT, r"$\mathrm{Y_{IRF}}$", C_IRF, dx=0.13, dy=-0.05)
label3(ax, AXLEN * ZHAT, r"$\mathrm{Z_{IRF}}$", C_IRF, dy=0.09)

# source vector + equatorial projection
uvO, uvF = VIEW.P([0, 0, 0])[0], VIEW.P(0.99 * foot_I)[0]
ax.plot(
    [uvO[0], uvF[0]], [uvO[1], uvF[1]], color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=4
)
arrow3(ax, [0, 0, 0], SRCLEN * r_I, C_SRC, ls="--", lw=1.5)
ax.plot(*VIEW.P(r_I)[0], "o", color="black", ms=3, zorder=7)
label3(ax, SRCLEN * r_I, r"$S\,(\theta,\phi)$", C_SRC, dx=0.16, dy=0.10, fs=13.5)

# LTP tangent frame (blue): N towards +Z_IRF, D = -S (right-handed NED)
P = r_I
BL, GL = 0.48, 0.38  # blue / green tangent-arrow lengths
arrow3(ax, P, P + BL * nLTP_I, C_LTP, lw=1.5, zorder=7)
arrow3(ax, P, P + BL * eLTP_I, C_LTP, lw=1.5, zorder=7)
arrow3(ax, P, P - BL * r_I, C_LTP, lw=1.0, zorder=7)
label3(ax, P + 0.30 * nLTP_I, r"$\mathrm{N_{ILTP}}$", C_LTP, dx=-0.12, dy=0.22)
label3(ax, P + BL * eLTP_I, r"$\mathrm{E_{ILTP}}$", C_LTP, dx=0.11, dy=0.02)
label3(ax, P - BL * r_I, r"$\mathrm{D_{ILTP}}$", C_LTP, dx=0.06, dy=-0.05)

# IAU tangent frame repeated (green) to show the psi_0 offset
arrow3(ax, P, P + GL * nIAU_I, C_IAU, lw=1.5, zorder=7)
arrow3(ax, P, P + GL * eIAU_I, C_IAU, lw=1.5, zorder=7)
label3(ax, P + GL * nIAU_I, r"$\mathrm{N_{IAU}}$", C_IAU, dx=-0.02, dy=0.03, fs=11.5)
label3(ax, P + GL * eIAU_I, r"$\mathrm{E_{IAU}}$", C_IAU, dx=0.11, dy=-0.03, fs=11.5)

# polarization + psi_LTP and psi_0
arrow3(
    ax,
    P - 0.32 * p_I,
    P + 0.44 * p_I,
    C_POL,
    ls="-",
    lw=1.5,
    mutation=12,
    zorder=6,
    double_sided=True,
)
psi_arc(
    ax,
    P,
    nLTP_I,
    p_I - 0.1,
    0.30,
    C_LTP,
    r"$\psi_{\mathrm{ILTP}}$",
    lab_r=1.73,
    t_frac=0.9,
)
# psi_arc(ax, P, nLTP_I, nIAU_I, 0.20, "0.2", r"$\psi_0$", lab_r=1.50,
#         t_frac=0.50, fs=11)

# ----------------------------------------------------------------------
fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, wspace=0.001)
fig.savefig("PolPy_Coord_Frames.pdf", bbox_inches="tight")
print("saved PolPy_Coord_Frames.pdf")


# ======================================================================
# 6. 2D Projected View of the Tangent Plane (IAU vs LTP)
# ======================================================================
fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_xlim(-1.25, 1.25)
ax2.set_ylim(-1.25, 1.25)


def vec2d(ang_deg):
    """Angle in degrees measured from North (Up) towards East (Left)."""
    rad = np.radians(ang_deg)
    return np.array([-np.sin(rad), np.cos(rad)])


# Define bases: N_IAU is Up(0), E_IAU is Left(90)
rot = 90 - PSI_0
N_IAU2 = vec2d(0 - rot)
E_IAU2 = vec2d(90 - rot)
P2 = vec2d(PSI_IAU - rot)

# N_LTP is offset by -PSI_0 so that N_IAU is at +PSI_0 relative to N_LTP
N_LTP2 = vec2d(-PSI_0 - rot)
E_LTP2 = vec2d(90 - PSI_0 - rot)

O2 = np.array([0.0, 0.0])


def arrow2d(ax, p0, p1, color, lw=1.5, ls="-", mutation=14, double_sided=False):
    if double_sided:
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict(
                arrowstyle="<|-|>",
                color=color,
                lw=lw,
                linestyle=ls,
                mutation_scale=mutation,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=4,
        )
    else:
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                linestyle=ls,
                mutation_scale=mutation,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=4,
        )


def arc2d_label(ax, ang0, ang1, radius, color, text, label_offset=1.2, fs=13):
    t = np.linspace(ang0, ang1, 45)
    pts = np.array([vec2d(a) for a in t]) * radius
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, zorder=3)
    # Arrowhead at the end (pointing in direction of increasing angle)
    arrow2d(ax, pts[-3], pts[-1], color, lw=1.5, mutation=10)
    # Label
    mid_ang = (ang0 + ang1) / 2
    mid_pt = vec2d(mid_ang) * (radius * label_offset)
    ax.text(
        mid_pt[0],
        mid_pt[1],
        text,
        color=color,
        fontsize=fs,
        ha="center",
        va="center",
        path_effects=WHITE_STROKE,
        zorder=5,
    )


# 1. Draw IAU frame (Green)
arrow2d(ax2, O2, N_IAU2, C_IAU)
arrow2d(ax2, O2, E_IAU2, C_IAU)
ax2.text(
    *(N_IAU2 * 1.1),
    r"$\mathrm{N_{IAU}}$",
    color=C_IAU,
    ha="center",
    va="center",
    fontsize=15,
    path_effects=WHITE_STROKE,
)
ax2.text(
    *(E_IAU2 * 1.06),
    r"$\mathrm{E_{IAU}}$",
    color=C_IAU,
    ha="center",
    va="center",
    fontsize=15,
    path_effects=WHITE_STROKE,
)

# 2. Draw LTP frame (Blue)
arrow2d(ax2, O2, N_LTP2, C_LTP)
arrow2d(ax2, O2, E_LTP2, C_LTP)
ax2.text(
    *(N_LTP2 * 1.12),
    r"$\mathrm{N_{ILTP}}$",
    color=C_LTP,
    ha="center",
    va="center",
    fontsize=15,
    path_effects=WHITE_STROKE,
)
ax2.text(
    *(E_LTP2 * 1.05),
    r"$\mathrm{E_{ILTP}}$",
    color=C_LTP,
    ha="center",
    va="center",
    fontsize=15,
    path_effects=WHITE_STROKE,
)

# 3. Draw Polarization vector (Orange dashed)
arrow2d(ax2, -P2 * 0.65, P2 * 0.75, C_POL, ls="-", lw=1.5, double_sided=True)
# arrow2d(ax2, P2 * 0.85, -P2 * 0.4, C_POL, ls="--", lw=2.2)

# 4. Draw Arcs
arc2d_label(
    ax2, -rot, PSI_IAU - rot, 0.4, C_IAU, r"$\psi_{\mathrm{IAU}}$", label_offset=1.25
)
arc2d_label(
    ax2,
    -PSI_0 - rot,
    -PSI_0 + PSI_LTP - rot,
    0.65,
    C_LTP,
    r"$\psi_{\mathrm{ILTP}}$",
    label_offset=1.2,
)
arc2d_label(ax2, -PSI_0 - rot, -rot, 0.25, "0.25", r"$\psi_0$", label_offset=1.3)

# Central dot
ax2.plot(0, 0, "o", color="black", ms=5, zorder=6)

fig.subplots_adjust(0.01, 0.01, 0.95, 0.95, wspace=0.001)
fig2.savefig("LTP_projection.pdf", bbox_inches="tight")
print("saved LTP_projection.pdf")
