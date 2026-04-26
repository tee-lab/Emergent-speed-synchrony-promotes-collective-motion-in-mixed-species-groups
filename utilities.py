import glob
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, EllipseCollection
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import SmoothBivariateSpline
from tqdm.notebook import trange

import sys
# sys.path.append('/Users/nabeel/Documents/Research/Fish Data Analysis/fish-data-analysis')
from src.polyfit import Poly2D, PolyFit2D
from sklearn.linear_model import ridge_regression


def load_dataset(files: str, n_fish: int = 15, discard=0):
    pos, vel = np.empty((0, n_fish, 2)), np.empty((0, n_fish, 2))
    valid = np.empty((0, n_fish), dtype=bool)

    for file in files:
        print(file)
        data = np.load(file)
        shape = data['pos'].shape
        if shape[1] < n_fish:
            pos_ = np.concatenate([data['pos'], np.full((shape[0], n_fish - shape[1], shape[2]), np.nan)], axis=1)
            vel_ = np.concatenate([data['vel'], np.full((shape[0], n_fish - shape[1], shape[2]), np.nan)], axis=1)
            valid_ = np.concatenate([data['valid'], np.full((shape[0], n_fish - shape[1], shape[2]), False)], axis=1)
        else:
            pos_ = data['pos']
            vel_ = data['vel']
            valid_ = data['valid']
        pos = np.concatenate((pos, pos_[discard:, ...]), axis=0)
        vel = np.concatenate((vel, vel_[discard:, ...]), axis=0)
        valid = np.concatenate((valid, valid_[discard:, ...]), axis=0)

    return pos, vel, valid


def compute_parameters(pos, vel):
    speed = np.linalg.norm(vel, axis=2)         # Individual speeds
    m = np.nanmean((vel.T / speed.T), axis=1)   # Polarization (vector)
    modm = np.linalg.norm(m, axis=0)            # Polarization magnitude
    v = np.nanmean(speed, axis=1)               # Average individual speed
    r = np.nanmean(np.linalg.norm(pos, axis=2), axis=1)  # Distance from center

    return m, modm, speed, v, r


def stratify_parameters(m, speed, r, R=35, center=0.5, boundary=0.5):
    speed_c, speed_b = speed.copy(), speed.copy()
    m_c, m_b = m.copy(), m.copy()

    speed_c[r > center * R] = np.nan
    m_c[:, r > center * R] = np.nan
    modm_c = np.linalg.norm(m_c, axis=0)
    v_c = np.nanmean(speed_c, axis=1)

    speed_b[r <= boundary * R] = np.nan
    m_b[:, r <= boundary * R] = np.nan
    modm_b = np.linalg.norm(m_b, axis=0)
    v_b = np.nanmean(speed_b, axis=1)

    return m_c, modm_c, speed_c, v_c, m_b, modm_b, speed_b, v_b


def compute_derivatives(m, modm, v, dt=1/25., DT=1):
    m_para = m / modm
    m_perp = np.stack((-m_para[1, :], m_para[0, :]))
    dm = (m[:, DT:] - m[:, :-DT]) / (DT * dt)
    dm = np.concatenate((dm, np.full((2, DT), np.nan)), axis=1)
    # dm = np.insert(dm, -1, np.nan, axis=1)
    dv = (v[DT:] - v[:-DT]) / (DT * dt)
    dv = np.concatenate((dv, np.full(DT, np.nan)))
    # dv = np.insert(dv, -1, np.nan)
    dm_para = np.einsum('ij,ij->j', dm, m_para)
    dm_perp = np.einsum('ij,ij->j', dm, m_perp)

    return dm, dm_para, dm_perp, dv

def compute_diffusion_3pt(m, modm, v, dt=1/25., DT=1):
    """ Compute 3-point estimate of diffusion. """
    m_para = m / modm
    m_perp = np.stack((-m_para[1, :], m_para[0, :]))
    dm3 = (m[:, 2 * DT:] + m[:, :-2 * DT] - 2 * m[:, DT:-DT])
    dm3 = np.concatenate((np.full((2, DT), np.nan), dm3, np.full((2, DT), np.nan)), axis=1)
    gm = np.einsum('ik,jk->ijk', dm3, dm3) / (2 * DT * dt)
    gm_para = np.einsum('ik,ijk,jk->k', m_para, gm, m_para)
    gm_perp = np.einsum('ik,ijk,jk->k', m_perp, gm, m_perp)

    dv3 = (v[2 * DT:] + v[:-2 * DT] - 2 * v[DT:-DT])
    dv3 = np.concatenate((np.full(DT, np.nan), dv3, np.full(DT, np.nan)))
    gv = dv3 ** 2 / (2 * DT * dt)

    return gm, gm_para, gm_perp, gv

def get_binned_estimates(dm_para, dv, modm, v, n_mbins=50, n_vbins=50, vscale=30):
    mbins = np.linspace(0, 1, n_mbins)
    vbins = np.linspace(0, vscale, n_vbins)

    m_bin_id = np.digitize(modm, mbins)
    v_bin_id = np.digitize(v, vbins)

    dm_binned = np.empty((n_mbins, n_vbins)) * np.nan
    dv_binned = np.empty((n_mbins, n_vbins)) * np.nan
    n_pts = np.zeros((n_mbins, n_vbins))

    for i in range(n_mbins):
        for j in range(n_vbins):
            dm_binned[i, j] = np.nanmean(dm_para[(m_bin_id == i) & (v_bin_id == j)])
            dv_binned[i, j] = np.nanmean(dv[(m_bin_id == i) & (v_bin_id == j)])
            n_pts[i, j] = np.sum((m_bin_id == i) & (v_bin_id == j))

    return mbins, vbins, dm_binned, dv_binned, n_pts

def get_binned_diffusion_estimates(gm_para, gm_perp, gv, modm, v, vscale, n_mbins=50, n_vbins=50):
    mbins = np.linspace(0, 1, n_mbins)
    vbins = np.linspace(0, vscale, n_vbins)

    m_bin_id = np.digitize(modm, mbins)
    v_bin_id = np.digitize(v, vbins)

    gm_para_binned = np.full((n_mbins, n_vbins), np.nan)
    gm_perp_binned = np.full((n_mbins, n_vbins), np.nan)
    gv_binned = np.full((n_mbins, n_vbins), np.nan)
    n_pts = np.zeros((n_mbins, n_vbins))   

    for i in range(n_mbins):
        for j in range(n_vbins):
            gm_para_binned[i, j] = np.nanmean(gm_para[(m_bin_id == i) & (v_bin_id == j)])
            gm_perp_binned[i, j] = np.nanmean(gm_perp[(m_bin_id == i) & (v_bin_id == j)])
            gv_binned[i, j] = np.nanmean(gv[(m_bin_id == i) & (v_bin_id == j)])
            n_pts[i, j] = np.sum((m_bin_id == i) & (v_bin_id == j))

    return mbins, vbins, gm_para_binned, gm_perp_binned, gv_binned, n_pts


def get_binned_diffusion_estimates_old(dm_para, dv, modm, v, n_mbins=50, n_vbins=50, vscale=30):
    mbins = np.linspace(0, 1, n_mbins)
    vbins = np.linspace(0, vscale, n_vbins)

    m_bin_id = np.digitize(modm, mbins)
    v_bin_id = np.digitize(v, vbins)

    Gm_binned = np.empty((n_mbins, n_vbins)) * np.nan
    Gv_binned = np.empty((n_mbins, n_vbins)) * np.nan
    Gmv_binned = np.empty((n_mbins, n_vbins)) * np.nan

    for i in range(n_mbins):
        for j in range(n_vbins):
            dm_bin = np.nanmean(dm_para[(m_bin_id == i) & (v_bin_id == j)])
            dv_bin = np.nanmean(dv[(m_bin_id == i) & (v_bin_id == j)])

            Gm_binned[i, j] = np.nanmean((dm_para[(m_bin_id == i) & (v_bin_id == j)] - dm_bin) ** 2)
            Gv_binned[i, j] = np.nanmean((dv[(m_bin_id == i) & (v_bin_id == j)] - dv_bin) ** 2)
            Gmv_binned[i, j] = np.nanmean(
                (dm_para[(m_bin_id == i) & (v_bin_id == j)] - dm_bin) *
                (dv[(m_bin_id == i) & (v_bin_id == j)] - dv_bin))

    return mbins, vbins, Gm_binned, Gv_binned, Gmv_binned

def get_binned_estimates_m_only(dm_para, modm, nbins=50):
    mbins = np.linspace(0, 1, nbins)
    m_bin_id = np.digitize(modm, mbins, right=True)

    dm_binned = np.full(nbins, np.nan)
    Gm_binned = np.full(nbins, np.nan)

    for i in range(nbins):
        dm_binned[i] = np.nanmean(dm_para[m_bin_id == i])
        Gm_binned[i] = np.nanvar(dm_para[m_bin_id == i])

    return mbins, dm_binned, Gm_binned


def get_binned_diffusion_estimates_m_only(dm_para, modm, nbins=50):
    mbins = np.linspace(0, 1, nbins)
    m_bin_id = np.digitize(modm, mbins)

    dm_binned = np.full(nbins, np.nan)

    for i in range(nbins):
        dm_binned[i] = np.nanvar(dm_para[m_bin_id == i])

    return mbins, dm_binned


def smooth(data, sigma=1, odd=False):
    data_mask = np.isnan(data)
    data[data_mask] = 0
    if odd:
        data = np.concatenate([-np.flipud(data), data], axis=0)
    else:
        data = np.concatenate([np.flipud(data), data], axis=0)
    data_smoothed = gaussian_filter(data, sigma=sigma)

    data = data[data.shape[0] // 2:]
    data_smoothed = data_smoothed[data_smoothed.shape[0] // 2:]
    data[data_mask] = np.nan
    # data_smoothed[data_mask] = np.nan

    return data_smoothed

def smooth_1d(data, sigma=1, odd=False):
    data_mask = np.isnan(data)
    data[data_mask] = 0
    if odd:
        data = np.concatenate([-np.flipud(data), data], axis=0)
    else:
        data = np.concatenate([np.flipud(data), data], axis=0)

    data_smoothed = gaussian_filter1d(data, sigma)
    
    data = data[data.shape[0] // 2:]
    data_smoothed = data_smoothed[data_smoothed.shape[0] // 2:]
    data[data_mask] = np.nan
    
    return data_smoothed

def plot_diffusion_field(ax, mbins, vbins, Gm, Gv, mscale, vscale, subsample=5):
    Gm_plot = Gm[::subsample, ::subsample].flatten()
    Gv_plot = Gv[::subsample, ::subsample].flatten()

    mm, vv = np.meshgrid(mbins[::subsample], vbins[::subsample])
    mv = np.column_stack((mm.ravel(), vv.ravel()))

    ec = EllipseCollection(
    widths=Gm_plot * mscale * subsample,
    heights=Gv_plot * vscale * subsample,
    offsets=mv, offset_transform=ax.transData,
    angles=0,
    # edgecolors=plt.cm.BuPu(plt.Normalize()(np.sqrt(Gm_plot ** 2 + Gv_plot ** 2))),
    # facecolors='w',
    linewidths=3.5,
    cmap='BuPu',
    # units='xy',
    )
    # ec.set_array(np.sqrt(Gm_plot ** 2 + Gv_plot ** 2))
    ax.set(xlabel='$m$', ylabel='$s$')
    ax.add_collection(ec)


def plot_derivative_fields(mbins, vbins, modm, v, dm, dv, dm_bounds=(-0.5, 0.5), dv_bounds=(-0.5, 0.5), quiver_scale=10):
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    ax[0].pcolormesh(mbins, vbins, dm.T, cmap='RdBu', vmin=dm_bounds[0], vmax=dm_bounds[1], rasterized=True)
    ax[0].set(title='$f_m(m, v)$', xlabel='$m$', ylabel='$v$')
    ax[1].pcolormesh(mbins, vbins, dv.T, cmap='RdBu', vmin=dv_bounds[0], vmax=dv_bounds[1], rasterized=True)
    ax[1].set(title='$f_v(m, v)$', xlabel='$m$', ylabel='$v$')

    ax[2].contour(mbins, vbins, dm.T, levels=[0], colors='b', linewidths=3)
    ax[2].contour(mbins, vbins, dv.T, levels=[0], colors='r', linewidths=3)
    ax[2].set(title='Nullclines (blue: $m$, red: $v$)', xlabel='$m$', ylabel='$v$')

    # ax[2].hist2d(modm, v, bins=(mbins, vbins), cmap='viridis', rasterized=True)
    # ax[2].quiver(mbins[::2], vbins[::2], dm[::2, ::2].T, dv[::2, ::2].T, scale=quiver_scale, angles='xy', scale_units='xy', width=2,
    #              color='w', units='dots')
    # ax[2].set(title='Flow field', xlabel='$m$', ylabel='$v$')
    ax[3].hist2d(modm, v, bins=(mbins, vbins), cmap='viridis', rasterized=True)
    ax[3].streamplot(mbins, vbins, dm.T, dv.T, linewidth=1, density=1.5, color='w')
    ax[3].set(title='Flow field', xlabel='$m$', ylabel='$v$')
    plt.tight_layout()
    plt.show()

def plot_diffusion_functions(mbins, vbins, vscale, Gm_para, Gm_perp, Gv, levels=100, mvmin=None, mvmax=None, vvmin=None, vvmax=None): #meshgrid=False):
    # if meshgrid:
    #     mbins, vbins = np.meshgrid(mbins, vbins)
    fig, ax = plt.subplots(1, 5, figsize=(20, 5), width_ratios=(1, 1, 0.05, 1, 0.05), layout='constrained')
    ax_gm_para, ax_gm_perp, ax_gv = ax[0], ax[1], ax[3]
    ax_cm, ax_cv = ax[2], ax[4]

    # vmax = np.max((Gm_para(mbins, vbins / vscale).max(), Gm_perp(mbins, vbins / vscale).max()))
    # vmin = np.min((Gm_para(mbins, vbins / vscale).min(), Gm_perp(mbins, vbins / vscale).min()))
    cm = ax_gm_para.contourf(mbins.ravel(), vbins.ravel(), Gm_para(mbins, vbins / vscale).T, levels=levels, vmin=mvmin, vmax=mvmax)
    ax_gm_perp.contourf(mbins.ravel(), vbins.ravel(), Gm_perp(mbins, vbins / vscale).T, levels=levels, vmin=mvmin, vmax=mvmax)
    cv = ax_gv.contourf(mbins.ravel(), vbins.ravel(), Gv(mbins, vbins / vscale).T * vscale ** 2, levels=levels, vmin=vvmin, vmax=vvmax)
    fig.colorbar(cm, cax=ax_cm)
    fig.colorbar(cv, cax=ax_cv)

    ax_gm_para.set(title='$G_m^\\|$', xlabel='$m$', ylabel='$v$')
    ax_gm_perp.set(title='$G_m^{\\perp}$', xlabel='$m$', ylabel='$v$')
    ax_gv.set(title='$G_v$', xlabel='$m$', ylabel='$v$')
    # plt.colorbar()
    # plt.title('$G_m(m, v)$ ($\\leftrightarrow$)')
    plt.show()

def plot_comparison_fields(mbins, vbins, modm_b, v_b, dm_b, dv_b, 
                           modm_c, v_c, dm_c, dv_c, quiver_scale=10):
    dist_c = np.histogram2d(modm_c, v_c, bins=(mbins, vbins), density=True)[0]
    dist_b = np.histogram2d(modm_b, v_b, bins=(mbins, vbins), density=True)[0]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    ax[0].pcolormesh(mbins, vbins, dist_c.T, cmap='viridis',)
    ax[1].pcolormesh(mbins, vbins, dist_b.T, cmap='viridis',)

    bound = np.nanmax(np.abs((dist_b - dist_c)))
    ax[2].pcolormesh(mbins, vbins, dist_b.T - dist_c.T, cmap='RdBu', vmin=-bound, vmax=bound)

    ax[0].quiver(mbins[::2], vbins[::2], dm_c[::2, ::2].T, dv_c[::2, ::2].T, scale=quiver_scale, angles='xy', scale_units='xy', width=2, units='dots', color='w')
    ax[1].quiver(mbins[::2], vbins[::2], dm_b[::2, ::2].T, dv_b[::2, ::2].T, scale=quiver_scale, angles='xy', scale_units='xy', width=2, units='dots', color='w')
    ax[2].quiver(mbins[::2], vbins[::2], dm_b[::2, ::2].T - dm_c[::2, ::2].T, dv_b[::2, ::2].T - dv_c[::2, ::2].T, 
                 scale=quiver_scale, angles='xy', scale_units='xy', width=2, units='dots')
    
    ax[0].set(title='Center', xlabel='$m$', ylabel='$v$')
    ax[1].set(title='Boundary', xlabel='$m$', ylabel='$v$')
    ax[2].set(title='Difference (B - C)', xlabel='$m$', ylabel='$v$')
    plt.tight_layout()
    plt.show()

def plot_comparison_curves(mbins, vbins, 
                           modm_b, v_b, dm_b, dv_b,
                           modm_c, v_c, dm_c, dv_c):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    ax[0, 0].hist2d(modm_c, v_c, bins=(mbins, vbins), cmap='viridis', rasterized=True)
    ax[0, 1].hist2d(modm_b, v_b, bins=(mbins, vbins), cmap='viridis', rasterized=True)
    ax[0, 0].streamplot(mbins, vbins, dm_c.T, dv_c.T, linewidth=1, density=1.5, color='w')
    ax[0, 1].streamplot(mbins, vbins, dm_b.T, dv_b.T, linewidth=1, density=1.5, color='w')
    ax[1, 0].contour(mbins, vbins, dm_c.T, levels=[0], colors='b', linewidths=3)
    ax[1, 0].contour(mbins, vbins, dv_c.T, levels=[0], colors='r', linewidths=3)
    ax[1, 1].contour(mbins, vbins, dm_b.T, levels=[0], colors='b', linewidths=3)
    ax[1, 1].contour(mbins, vbins, dv_b.T, levels=[0], colors='r', linewidths=3)

    ax[0, 0].set(title='Flow field - Center', ylabel='$v$')
    ax[0, 1].set(title='Flow field - Boundary')
    ax[1, 0].set(title='Nullclines - Center', xlabel='$m$', ylabel='$v$')
    ax[1, 1].set(title='Nullclines - Boundary', xlabel='$m$')

    plt.tight_layout()
    plt.show()

def plot_diffusion_fields(mbins, vbins, modm, v, Gm, Gv, Gmv, Gm_max=1, Gv_max=1000, Gmv_bounds=(-100, 100)):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].pcolormesh(mbins, vbins, Gm.T, vmin=0, vmax=Gm_max, rasterized=True)
    ax[0].set(title='$G_m(m, v)$', xlabel='$m$', ylabel='$v$')
    ax[1].pcolormesh(mbins, vbins, Gv.T, vmin=0, vmax=Gv_max, rasterized=True)
    ax[1].set(title='$G_v(m, v)$', xlabel='$m$', ylabel='$v$')
    ax[2].pcolormesh(mbins, vbins, Gmv.T, vmin=Gmv_bounds[0], vmax=Gmv_bounds[1], rasterized=True, cmap='RdBu')
    ax[2].set(title='$G_{mv}(m, v)$', xlabel='$m$', ylabel='$v$')

    plt.tight_layout()
    plt.show()

def fit_drift_splines(modm, v, dm_para, dv, vscale, fmkwargs, fvkwargs):
    """ Fit drift functions using smoothing splines. Does not work with long time series. """
    nan_idx = np.isnan(modm) | np.isnan(v) | np.isnan(dm_para) | np.isnan(dv)    
# modm_ = modm[~nan_idx]
# v_ = (1 / vscale) * v[~nan_idx]
# dm_ = dm_para[~nan_idx]
# dv_ = (1 / vscale) * dv[~nan_idx]

    modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx], ])#-modm[~nan_idx], modm[~nan_idx]])
    v_= (1 / vscale) * np.concatenate([v[~nan_idx], v[~nan_idx], ])
    dm_ = np.concatenate([-dm_para[~nan_idx], dm_para[~nan_idx], ])
    dv_ = (1 / vscale) * np.concatenate([dv[~nan_idx], dv[~nan_idx], ])#dv[~nan_idx], dv[~nan_idx]])
    fm_sbs = SmoothBivariateSpline(modm_, v_, dm_, **fmkwargs)
    fv_sbs = SmoothBivariateSpline(modm_, v_, dv_, **fvkwargs)
    return fm_sbs, fv_sbs

def fit_diffusion_splines(modm, v, gm_para, gm_perp, gv, vscale, gmkwargs, gvkwargs):
    nan_idx = np.isnan(modm) | np.isnan(v) | np.isnan(gm_para) | np.isnan(gm_perp) | np.isnan(gv)
    modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx],])# -modm[~nan_idx], modm[~nan_idx]])
    v_ = (1 / vscale) *np.concatenate([v[~nan_idx], v[~nan_idx],])# -v[~nan_idx], -v[~nan_idx]]) / vscale

    gm_para_ = np.concatenate([gm_para[~nan_idx], gm_para[~nan_idx],])# gm_para[~nan_idx], gm_para[~nan_idx]])
    gm_perp_ = np.concatenate([gm_perp[~nan_idx], gm_perp[~nan_idx],])# gm_perp[~nan_idx], gm_perp[~nan_idx]])
    gv_ = (1 / vscale ** 2) * np.concatenate([gv[~nan_idx], gv[~nan_idx], ])

    Gm_para_sbs = SmoothBivariateSpline(modm_, v_, gm_para_, **gmkwargs)#kx=4, ky=2)
    Gm_perp_sbs = SmoothBivariateSpline(modm_, v_, gm_perp_, **gmkwargs)#kx=4, ky=2)
    Gv_sbs = SmoothBivariateSpline(modm_, v_, gv_, **gvkwargs)

    return Gm_para_sbs, Gm_perp_sbs, Gv_sbs

def fit_drift_splines_from_bins(mbins, vbins, dm_binned, dv_binned, n_pts, vscale, fmkwargs, fvkwargs):
    mm, vv = np.meshgrid(mbins, vbins)
    m_, v_ = mm.T.ravel(), vv.T.ravel() / vscale
    dm_, dv_ = dm_binned.ravel(), dv_binned.ravel() / vscale
    # w = n_pts.ravel()
    m_ = np.concatenate([mm.T.ravel(), -mm.T.ravel()])
    v_ = np.concatenate([vv.T.ravel(), vv.T.ravel()]) / vscale

    dm_ = np.concatenate([dm_binned.ravel(), -dm_binned.ravel()])
    dv_ = np.concatenate([dv_binned.ravel(), dv_binned.ravel()]) / vscale
    w = np.concatenate([n_pts.ravel(), n_pts.ravel()])
    # w /= 

    nan_idx = (w == 0)
    m_, v_ = m_[~nan_idx], v_[~nan_idx]
    dm_, dv_, w = dm_[~nan_idx], dv_[~nan_idx], w[~nan_idx]

    fm_sbs = SmoothBivariateSpline(m_, v_, dm_, w=np.sqrt(w), **fmkwargs)
    fv_sbs = SmoothBivariateSpline(m_, v_, dv_, w=np.sqrt(w), **fvkwargs)

    return fm_sbs, fv_sbs

def fit_diffusion_splines_from_bins(mbins, vbins, gm_para_binned, gm_perp_binned, gv_binned, n_pts, vscale, fmkwargs, fvkwargs):
    mm, vv = np.meshgrid(mbins, vbins)
    m_, v_ = mm.T.ravel(), vv.T.ravel() / vscale
    gm_para_, gm_perp_, gv_ = gm_para_binned.ravel(), gm_perp_binned.ravel(), gv_binned.ravel() / vscale
    # w = n_pts.ravel()
    m_ = np.concatenate([mm.T.ravel(), -mm.T.ravel()])
    v_ = np.concatenate([vv.T.ravel(), vv.T.ravel()]) / vscale

    gm_para_ = np.concatenate([gm_para_binned.ravel(), -gm_para_binned.ravel()])
    gm_perp_ = np.concatenate([gm_perp_binned.ravel(), -gm_perp_binned.ravel()])
    gv_ = np.concatenate([gv_binned.ravel(), gv_binned.ravel()]) / vscale
    w = np.concatenate([n_pts.ravel(), n_pts.ravel()])
    # w /= 

    nan_idx = (w == 0)
    m_, v_ = m_[~nan_idx], v_[~nan_idx]
    gm_para_, gm_perp_ = gm_para_[~nan_idx], gm_perp_[~nan_idx]
    gv_, w = gv_[~nan_idx], w[~nan_idx]

    gm_para_sbs = SmoothBivariateSpline(m_, v_, gm_para_, w=np.sqrt(w), **fmkwargs)
    gm_perp_sbs = SmoothBivariateSpline(m_, v_, gm_perp_, w=np.sqrt(w), **fmkwargs)
    gv_sbs = SmoothBivariateSpline(m_, v_, gv_, w=np.sqrt(w), **fvkwargs)

    return gm_para_sbs, gm_perp_sbs, gv_sbs

def fit_drift_functions(modm, v, dm_para, dv, vscale, fmkwargs, fvkwargs):
    nan_idx = np.isnan(modm) | np.isnan(v) | np.isnan(dm_para) | np.isnan(dv)
    
    modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx], ])#-modm[~nan_idx], modm[~nan_idx]])
    v_= (1 / vscale) * np.concatenate([v[~nan_idx], v[~nan_idx], ])#-v[~nan_idx], -v[~nan_idx]])
    
    # fm is an odd function of modm.
    dm_ = np.concatenate([-dm_para[~nan_idx], dm_para[~nan_idx], ])#-dm_para[~nan_idx], dm_para[~nan_idx]])
    fitter = PolyFit2D(xlabel='m', ylabel='v',
                       threshold=fmkwargs['threshold'], 
                       xdegree=fmkwargs['mdegree'], 
                       ydegree=fmkwargs['vdegree'], 
                       alpha=fmkwargs['alpha'])
    fm = fitter.fit(np.array([modm_, v_]).T, dm_)
    
    # fv is an even function of modm.
    dv_ = (1 / vscale) * np.concatenate([dv[~nan_idx], dv[~nan_idx], ])#dv[~nan_idx], dv[~nan_idx]])
    fitter = PolyFit2D(xlabel='m', ylabel='v',
                       threshold=fvkwargs['threshold'], 
                       xdegree=fvkwargs['mdegree'], 
                       ydegree=fvkwargs['vdegree'], 
                       alpha=fvkwargs['alpha'])
    fv = fitter.fit(np.array([modm_, v_]).T, dv_)

    return fm, fv

def fit_diffusion_functions(modm, v, gm_para, gm_perp, gv, vscale, gmkwargs, gvkwargs):
    nan_idx = np.isnan(modm) | np.isnan(v) | np.isnan(gm_para) | np.isnan(gm_perp) | np.isnan(gv)
    modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx],])# -modm[~nan_idx], modm[~nan_idx]])
    v_ = (1 / vscale) *np.concatenate([v[~nan_idx], v[~nan_idx],])# -v[~nan_idx], -v[~nan_idx]]) / vscale

    gm_para_ = np.concatenate([gm_para[~nan_idx], gm_para[~nan_idx],])# gm_para[~nan_idx], gm_para[~nan_idx]])
    gm_perp_ = np.concatenate([gm_perp[~nan_idx], gm_perp[~nan_idx],])# gm_perp[~nan_idx], gm_perp[~nan_idx]])
    gv_ = (1 / vscale ** 2) * np.concatenate([gv[~nan_idx], gv[~nan_idx], ])
    
    fitter = PolyFit2D(xlabel='m', ylabel='v',
                       threshold=gmkwargs['threshold'], 
                       xdegree=gmkwargs['mdegree'], 
                       ydegree=gmkwargs['vdegree'], 
                       alpha=gmkwargs['alpha'])
    Gm_para = fitter.fit(np.stack([modm_, v_]).T, gm_para_)
    Gm_perp = fitter.fit(np.stack([modm_, v_]).T, gm_perp_)

    fitter = PolyFit2D(xlabel='m', ylabel='v',
                       threshold=gvkwargs['threshold'], 
                       xdegree=gvkwargs['mdegree'], 
                       ydegree=gvkwargs['vdegree'], 
                       alpha=gvkwargs['alpha'])
    
    Gv = fitter.fit(np.stack([modm_, v_]).T, gv_)
    
    return Gm_para, Gm_perp, Gv

    


def fit_drift_functions_1d(modm, dm_para, fmkwargs):
    nan_idx = np.isnan(modm) | np.isnan(dm_para)
    modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx]])
    dm_ = np.concatenate([-dm_para[~nan_idx], dm_para[~nan_idx]])


# def fit_drift_functions_from_model(modm, v, dm_para, dv, vscale):
#     """
#     fm(m, v) = a1 * m + a2 * m * v + a3 * m * v ** 2 + a4 * m ** 3 + a5 * m ** 3 * v
#     fv(m, v) = b1 + b2 * m ** 2 + b3 * v + b4 * m ** 2 * v + b5 * v ** 2 + b6 * v ** 3
#     """

#     nan_idx = np.isnan(modm) | np.isnan(v) | np.isnan(dm_para) | np.isnan(dv)
    
#     modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx]])
#     v_= np.concatenate([v[~nan_idx], v[~nan_idx]]) / vscale
    
#     fm_terms = np.array([modm_, 
#                      modm_ * v_, 
#                      modm_ * v_ ** 2, 
#                      modm_ ** 3,
#                      modm_ ** 3 * v_,
#                      modm_ ** 3 * v_ ** 2,
#     ]).T

#     fs_terms = np.array([np.ones_like(modm_),
#                         modm_ ** 2,
#                         v_,
#                         modm_ ** 2 * v_,
#                         v_ ** 2,
#                         v_ ** 3,
#     ]).T


#     # fm is an odd function of modm.
#     dm_ = np.concatenate([-dm_para[~nan_idx], dm_para[~nan_idx]])
#     a = ridge_regression(fm_terms, dm_, alpha=0)
#     fm = Poly2D(xdegree=3, ydegree=2, xlabel='m', ylabel='s',
#                 coeffs=[
#                     0, 0, 0,
#                     a[0], a[1], a[2],
#                     0, 0, 0,
#                     a[3], a[4], a[5],
#                 ])
        
#     # fv is an even function of m.
#     dv_ = np.concatenate([dv[~nan_idx], dv[~nan_idx]]) / vscale
#     b = ridge_regression(fs_terms, dv_, alpha=0)
#     fv = Poly2D(xdegree=2, ydegree=3, xlabel='m', ylabel='s',
#                 coeffs=[
#                     b[0], b[2], b[4], b[5],
#                     0, 0, 0, 0,
#                     b[1], b[3], 0, 0,
#                 ])
    
#     return fm, fv

def fit_drift_functions_from_model(modm, v, dm_para, dv, vscale):
    """ Fit a polynomial, guessed by visual examination of the nullclines. """
    nan_idx = np.isnan(modm) | np.isnan(v) | np.isnan(dm_para) | np.isnan(dv)
    
    modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx]])
    v_= np.concatenate([v[~nan_idx], v[~nan_idx]]) / vscale
    
    fm_terms = np.array([modm_, 
                     modm_ * v_, 
                    #  modm_ * v_ ** 2, 
                     modm_ ** 3 * v_,
    ]).T

    fs_terms = np.array([np.ones_like(modm_),
                        modm_ ** 2,
                        v_,
                        modm_ ** 2 * v_,
                        v_ ** 2,
                        v_ ** 3,
    ]).T


    # fm is an odd function of modm.
    dm_ = np.concatenate([-dm_para[~nan_idx], dm_para[~nan_idx]])
    a = ridge_regression(fm_terms, dm_, alpha=0)
    fm = Poly2D(xdegree=3, ydegree=2, xlabel='m', ylabel='s',
                coeffs=[
                    0, 0, 0,
                    a[0], a[1], 0,
                    0, 0, 0,
                    0, a[2], 0,
                ])
        
    # fv is an even function of m.
    dv_ = np.concatenate([dv[~nan_idx], dv[~nan_idx]]) / vscale
    b = ridge_regression(fs_terms, dv_, alpha=0)
    fv = Poly2D(xdegree=2, ydegree=3, xlabel='m', ylabel='s',
                coeffs=[
                    b[0], b[2], b[4], b[5],
                    0, 0, 0, 0,
                    b[1], b[3], 0, 0,
                ])
    
    return fm, fv

def fit_drift_functions_from_model_old(modm, v, dm_para, dv, vscale):
    """ Fit a model inspired by mean-field derivations in 1D. """
    nan_idx = np.isnan(modm) | np.isnan(v) | np.isnan(dm_para) | np.isnan(dv)
    
    modm_ = np.concatenate([-modm[~nan_idx], modm[~nan_idx]])
    v_= np.concatenate([v[~nan_idx], v[~nan_idx]]) / vscale
    
    fm_terms = np.array([modm_, 
                     modm_ * v_, 
                     modm_ * v_ ** 2, 
                     modm_ ** 3,
    ]).T

    fs_terms = np.array([np.ones_like(modm_),
                        modm_ ** 2,
                        v_,
                        modm_ ** 2 * v_,
                        v_ ** 2,
                        v_ ** 3,
    ]).T


    # fm is an odd function of modm.
    dm_ = np.concatenate([-dm_para[~nan_idx], dm_para[~nan_idx]])
    a = ridge_regression(fm_terms, dm_, alpha=0)
    fm = Poly2D(xdegree=3, ydegree=2, xlabel='m', ylabel='s',
                coeffs=[
                    0, 0, 0,
                    a[0], a[1], a[2],
                    0, 0, 0,
                    a[3], 0, 0,
                ])
        
    # fv is an even function of m.
    dv_ = np.concatenate([dv[~nan_idx], dv[~nan_idx]]) / vscale
    b = ridge_regression(fs_terms, dv_, alpha=0)
    fv = Poly2D(xdegree=2, ydegree=3, xlabel='m', ylabel='s',
                coeffs=[
                    b[0], b[2], b[4], b[5],
                    0, 0, 0, 0,
                    b[1], b[3], 0, 0,
                ])
    
    return fm, fv


# The following two functions are by dpsanders. 
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array

    Code from: https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width

    Code from: https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, joinstyle='miter', capstyle='round')
    
    # ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

def plot_trajectories(pos, m, v, vscale, start, end):
    n_fish = pos.shape[1]
    pos_group = np.nanmean(pos, axis=1)
    modm = np.linalg.norm(m, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    axt, axp = ax
    for i in range(n_fish):
        axt.plot(pos[start:end, i, 0], 
                pos[start:end, i, 1], lw=1, c='k', alpha=0.2, solid_joinstyle='miter')

    # axt.plot(gaussian_filter1d(pos_group[start:end, 0], 0.1), 
    #          gaussian_filter1d(pos_group[start:end, 1], 0.1), lw=3)
    # axp.plot(gaussian_filter1d(modm[start:end], 2), gaussian_filter1d(v[start:end], 2), lw=3)

    colorline(axt, 
            gaussian_filter1d(pos_group[start:end, 0], 2),
            gaussian_filter1d(pos_group[start:end, 1], 2),
            cmap=plt.get_cmap('YlOrRd'), linewidth=3)

    colorline(axp, gaussian_filter1d(modm[start:end], 2),
                gaussian_filter1d(v[start:end], 2), 
                cmap=plt.get_cmap('YlOrRd'), linewidth=3)

    axt.arrow(pos_group[end, 0], pos_group[end, 1], 
            10 * m[0, end], 10 * m[1, end], lw=3, joinstyle='miter', head_width=2, head_length=3, fc='k', ec='k', 
            aa=True, overhang=.3, zorder=3)

    axt.set(aspect='equal', xlim=[-36, 36], ylim=[-36, 36],)
    axt.set_axis_off()
    axp.set(xlim=[0, 1], ylim=[0, vscale], xlabel='$m$', ylabel='$v$')
    plt.tight_layout()
    plt.show()

def simulate_sde(fm, fv, Gm_para, Gm_perp, Gv, m0, v0, dt, T):
    rng = np.random.default_rng()
    warnings.filterwarnings('ignore', message='Conversion of an array with ndim > 0')
    # Note: Does simulation with non-dimensionalized velocity.
    Gm_para_ = lambda m, v: np.clip(Gm_para(m, v), 0.01, None)
    Gm_perp_ = lambda m, v: np.clip(Gm_perp(m, v), 0.01, None)
    Gv_ = lambda m, v: np.clip(Gv(m, v), .1, None)

    # Gm_para_ = lambda m, v: np.exp(Gm_para(m, v), )#0.0, None)
    # Gm_perp_ = lambda m, v: np.exp(Gm_perp(m, v), )#0.0, None)
    # Gv_ = lambda m, v: np.exp(Gv(m, v), )#0.0, None)

    m = np.empty((2, T))
    v = np.empty(T)
    m[:, 0] = m0
    v[0] = v0
    
    dW_m_para = rng.standard_normal(size=(T)) * np.sqrt(dt)
    dW_m_perp = rng.standard_normal(size=(T)) * np.sqrt(dt)
    dW_v = rng.standard_normal(size=(T)) * np.sqrt(dt)

    for t in range(T - 1):
        modm = np.linalg.norm(m[:, t])
        m_para = m[:, t] / modm
        m_perp = np.array((-m_para[1], m_para[0]))

        dm_para = dt * fm(modm, v[t]) + dW_m_para[t] * np.sqrt(Gm_para_(modm, v[t]))
        dm_perp = dW_m_perp[t] * np.sqrt(Gm_perp_(modm, v[t]))
        # print(f'{m[:, t]=}, {v[t]=}, {fm(m[:, t], v[t])=}')
        m[:, t + 1] = m[:, t] + dm_para * m_para + dm_perp * m_perp
        # print(np.linalg.norm(m[:, t + 1]))
        if np.linalg.norm(m[:, t + 1]) > 1:
            # print(f'Before: {np.linalg.norm(m[:, t + 1])}')
            # m[:, t + 1] = m[:, t + 1] / np.linalg.norm(m[:, t + 1])
            dm_para = dt * fm(modm, v[t]) - dW_m_para[t] * np.sqrt(Gm_para_(modm, v[t]))
            m[:, t + 1] = m[:, t] + dm_para * m_para + dm_perp * m_perp
            if np.linalg.norm(m[:, t + 1]) > 1:
                m[:, t + 1] = m[:, t + 1] / np.linalg.norm(m[:, t + 1])
        
        # print(f'{v[t], modm, fv(modm, v[t]), Gv_(modm, v[t])}')
        v[t + 1] = (v[t] + dt * fv(modm, v[t]) + dW_v[t] * np.sqrt(Gv_(modm, v[t])))
        if v[t + 1] < 0:
            v[t + 1] = (v[t] + dt * fv(modm, v[t]) - dW_v[t] * np.sqrt(Gv_(modm, v[t])))
        
    return m, v

def autocorr(x, lags=10000):
    if lags is None:
        lags = len(x)
    N = len(x)
    acf = np.empty(lags)
    x = (x - np.nanmean(x)) / np.nanstd(x)
    for i in trange(lags):
        acf[i] = np.nansum(x[i:] * x[:N-i])
    return acf / acf[0]

def plot_model_diagnostics(modm, v, modm_sim, v_sim, vscale, dt, dt_sim, lags):
    acf_m_act = autocorr(x=modm, lags=lags)
    acf_m_sim = autocorr(x=modm_sim[::int(dt / dt_sim)], lags=lags)

    acf_v_act = autocorr(x=v, lags=lags)
    acf_v_sim = autocorr(x=v_sim[::int(dt / dt_sim)], lags=lags)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    ax_m, ax_v = ax[0, 0], ax[1, 0]
    ax_mv_act, ax_mv_sim = ax[0, 2], ax[1, 2]
    ax_m_acf, ax_v_acf = ax[0, 1], ax[1, 1]

    ax_m.hist(modm, bins=100, alpha=0.5, density=True, label='Data', )#log=True)
    ax_m.hist(modm_sim, bins=100, alpha=0.5, density=True, label='SDE Model', )#log=True)

    ax_v.hist(v, bins=200, range=(0, 1.5 * vscale), alpha=0.5, density=True, )#log=True)
    ax_v.hist(vscale * v_sim, bins=200, range=(0, 1.5 * vscale), alpha=0.5, density=True, )#log=True)

    ax_mv_act.hist2d(modm, v, range=((0, 1), (0, 1.2 * vscale)), bins=100)
    ax_mv_sim.hist2d(modm_sim, vscale * v_sim, range=((0, 1), (0, 1.2 * vscale)), bins=100)

    ax_m_acf.plot(np.linspace(0, lags * dt, lags), acf_m_act, lw=4, label='Data')
    ax_m_acf.plot(np.linspace(0, lags * dt, lags), acf_m_sim, lw=4, label='SDE Model')
    ax_m_acf.axhline(0, lw=0.5, c='k')

    ax_v_acf.plot(np.linspace(0, lags * dt, lags), acf_v_act, lw=4, label='Data')
    ax_v_acf.plot(np.linspace(0, lags * dt, lags), acf_v_sim, lw=4, label='SDE Model')
    ax_v_acf.axhline(0, lw=0.5, c='k')

    ax_m_acf.set(xlabel='$\\tau$', ylabel='ACF $\\rho_m(\\tau)$', title='$m$ autocorrelation')
    ax_v_acf.set(xlabel='$\\tau$', ylabel='ACF $\\rho_v(\\tau)$', title='$v$ autocorrelation')
    ax_m_acf.legend()

    ax_m.set(xlabel='$m$', ylabel='Density', title='$m$ histograms')
    ax_v.set(xlabel='$v$', ylabel='Density', title='$v$ histograms')
    ax_m.legend()
    ax_mv_act.set(xlabel='$m$', ylabel='$v$', title='$(m, v)$ histogram – Data')
    ax_mv_sim.set(xlabel='$m$', ylabel='$v$', title='$(m, v)$ histogram – Model')
    plt.show()

def plot_noise_diagnostics(modm, v, vscale, dm_para, dm_perp, dv, fm, fv, Gm_para, Gm_perp, Gv):
    eta_m_para = (dm_para - fm(modm, v / vscale)) / np.sqrt(Gm_para(modm, v / vscale))
    eta_m_perp = dm_perp / np.sqrt(Gm_perp(modm, v / vscale))
    eta_v = (dv / vscale - fv(modm, v / vscale)) / np.sqrt(Gv(modm, v / vscale))

    acf_eta_m_para = autocorr(eta_m_para, 10)
    acf_eta_m_perp = autocorr(eta_m_perp, 10)
    acf_eta_v = autocorr(eta_v, 10)

    xplot = np.linspace(-10, 10, 101)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), layout='constrained', sharey='row')
    ax[0, 0].hist(eta_m_para, bins=100, range=(-10, 10), density=True)
    ax[0, 1].hist(eta_m_perp, bins=100, range=(-10, 10), density=True)
    ax[0, 2].hist(eta_v, bins=100, range=(-10, 10), density=True)
    for i in range(3):
        ax[0, i].plot(xplot, np.exp(- xplot ** 2 / (2 * 4)) / np.sqrt(np.pi * 2 * 4), lw=4, c='k')

    ax[0, 0].set(title='$m_\\|$ residual',xlabel='$\\eta_{m_\\|}$', ylabel='Density')
    ax[0, 1].set(title='$m_\\perp$ residual',xlabel='$\\eta_{m_\\perp}$')
    ax[0, 2].set(title='$v$ residual',xlabel='$\\eta_{v}$')

    ax[1, 0].plot(acf_eta_m_para, lw=4)
    ax[1, 0].axhline(0, c='k', lw=1)

    ax[1, 1].plot(acf_eta_m_perp, lw=4)
    ax[1, 1].axhline(0, c='k', lw=1)

    ax[1, 2].plot(acf_eta_v, lw=4)
    ax[1, 2].axhline(0, c='k', lw=1)

    ax[1, 0].set(ylabel='ACF')
    for i in range(3):
        ax[1, i].set(xlabel='Lag (timepoints)')

    plt.show()