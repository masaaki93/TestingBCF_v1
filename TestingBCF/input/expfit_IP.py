import numpy as np
import matplotlib.pyplot as plt
import os

import input.expfit as expfit

import jax
from spectral_density_fit import spectral_density_fitter
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import lsq_linear


# ----------------------------------------
#   Fitting by a sum of exponential functions
# ----------------------------------------
"""
Fitting \tilde{L}(ω) (data) using the spectral_density_fittter algorithm
"""
def IP_fit(Lω_filename, H_init, κ_init, g_init, Lω_data = np.array([]), fitlog = True, Lt_filename = None, dir = None, tol = 1e-8, opttol_rel = 1e-6):

    if len(Lω_data) == 0:
        Lω_data = np.loadtxt(Lω_filename)

    # data
    ω = Lω_data[:,0]
    dω = ω[1]-ω[0]
    Lω = Lω_data[:,1]

    # initial guess
    jax.config.update("jax_enable_x64", True)
    M = len(κ_init)
    opt = spectral_density_fitter(ω,Lω,M,fitlog=fitlog)
    ps = opt.Hκg_to_ps(H_init,κ_init,g_init)

    # plt.figure(figsize=(8, 6))
    # plt.plot(ω,Lω,color='black',lw=3,label='Data')
    # plt.plot(ω,opt.Jfun(ω,ps).squeeze(),color='red',lw=3,label='Initial guess')
    # plt.xlabel(r"$\omega$", fontsize=15)
    # plt.ylabel(r"$\tilde{L}(\omega)$", fontsize=15)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=15)
    # plt.tight_layout()
    # plt.show()

    # optimization
    opt.set_ftol_rel(opttol_rel)
    ps = opt.optimize(ps)
    H,κ,g = opt.ps_to_Hκg(ps)
    Heff = H-0.5j*np.diag(κ)

    λ, V = np.linalg.eig(Heff)
    V /= np.sqrt(np.sum(V**2,axis=0))
    G = g @ V

    d = G[0,:]**2 / (2*np.pi)
    z = 1j*λ

    # regularization
    d, z, K = expfit.regularization(d, z, tol)
    η = 0.

    # output
    if dir == None:
        filename = f'IP_K{K}'
    else:
        os.makedirs(dir, exist_ok=True)
        filename = dir + '/' + f'IP_K{K}'

    expfit.savefile(Lt_filename, Lω_filename, d, z, η, K, filename)

    return d, z, η

"""
Constructing the initial guess for 'IP_fit' from a model correlation function
    Lmod(t≧0) = Σ_{l=0}^{M-1} a[l] e^{- z[l] t}
using the method in the note (spectral_density_fit 5).
"""
def IP_initial_guess_naive(d, z):
    M = len(d)
    # b = np.abs(d)
    b = np.where(d.real >= 0.,
                    d.real,
                    1e-5)

    H_init = np.diag(z.imag)
    κ_init = 2 * z.real
    g_init = np.sqrt(2 * np.pi * b)

    return H_init, κ_init, g_init.reshape(1,-1)

def IP_initial_guess_peaks(Lω_filename, Mfloor):

    def Lorentzian(ω,ωi,κ,g):

        def Lorentzian_value(ω):
            if len(ωi) == 0:
                return 0.
            else:
                return np.sum(κ * g **2 / (2 * np.pi) / ((ω-ωi)**2 + κ**2/4))

        if isinstance(ω, (list, np.ndarray)):
            return np.vectorize(Lorentzian_value)(ω)
        else:
            return Lorentzian_value(ω)

    # data
    Lω_data = np.loadtxt(Lω_filename)
    ω = Lω_data[:,0]
    dω = ω[1]-ω[0]
    Lω = Lω_data[:,1]

    ωi = np.array([])
    κ = np.array([])
    g = np.array([])

    while True:
        peak_inds, peak_props = find_peaks(Lω-Lorentzian(ω,ωi,κ,g),width=1e-2,prominence=5e-4)
        ωi_new = ω[peak_inds]
        κ_new = dω * peak_props["widths"]

        ωi = np.append(ωi, ωi_new)
        κ = np.append(κ, κ_new)

        C = 1/(2 * np.pi) * κ[None,:] / ((ω[:,None]-ωi[None,:])**2 + κ[None,:]**2/4)
        res = lsq_linear(C, Lω, bounds=(0, np.inf))  # Nonnegative constraint
        g = np.sqrt(res.x)

        if len(ωi) >= Mfloor or len(ωi_new) == 0:
            break

    H_init = np.diag(ωi)
    κ_init = κ
    g_init = g

    return H_init, κ_init, g_init.reshape(1,-1)
