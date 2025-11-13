import numpy as np
import importlib
import os
import input.exact as exact
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from settings import SD, ω0v0

import time

### spectral density name
SD_name = SD()

### input data
module_name = f'{SD_name}.data'
data = importlib.import_module(module_name)

### parameters
ω0, v0 = ω0v0()
M = 1/(ω0*v0**2)   # Mass scale s.t. M = 1
β = data.get_β()     # inverse temperature  # β = -1 for the zero temperature
params = np.array([M, ω0, β])

### title
dir_name = SD_name + '/' + data.get_title()
os.makedirs(dir_name, exist_ok=True)
title = dir_name + '/exact'

### η
η = data.get_η()

### J & ωmax (these should be given if η == None)
if η == None:
    J = getattr(data, 'J_')  # we call J a lot for 'η_integral' and hence we should import data.J_ (not data.J, which may take more time to call)
    ωmax = 1e+1 # J(ω >= ωmax) = 0 is assumed
else:
    J = None; ωmax = None

##########################################
# plot J(ω)
##########################################

ω = np.linspace(0, 100, 500)
plt.figure(figsize=(8, 6))
plt.plot(ω, data.J(ω), color = 'black', lw = 2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\omega$', fontsize=15)
plt.xlabel(r'$J(\omega)$', fontsize=15)
plt.savefig(title + '_J.png', format='png')
plt.close()

#########################################
# q2p2
#########################################
## q2p2 from η_integral
tols = np.array([1e-6,1e-8,1e-10,1e-12])
START = time.time()
exact.q2p2_tols(params, tols, title, η = η, J = J, ωmax = ωmax)
END = time.time()
print(f'q2p2 computation time = {END-START} seconds')
print()

### Find v0
# q2, _ = exact.q2p2(params, tol = 1e-5, η = η, J = J, ωmax = ωmax)
# print(q2*v0**2)
# exit()

### q2p2 from ηt ('data.py' should include the function 'ηt')
# tf = 100; dt = 1e-2; t = np.linspace(0., tf, int(tf/dt)+1)  # time grid
# ηt = data.ηt(t)
# plt.figure(figsize=(8, 6))
# plt.plot(t, ηt, color = 'black', lw = 2)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel(r'$t$', fontsize=15)
# plt.ylabel(r'$\eta(t)$', fontsize=15)
# plt.savefig(title + '_ηt.png', format='png')
# plt.close()
# q2, p2 = exact.q2p2_ηt(params, t, ηt, tol = 1e-6)
# print('From η(t):')
# print(f'<q^2>_eq = {q2}')
# print(f'<p^2>_eq = {p2}')
# exit()

##########################################
# Cqq(t), Cpp(t)
##########################################
### Fourier region (max(|ω|) < ωmax must be satisfied when 'η_integral' is used)
ω = np.linspace(-5,15,300)

### CqqCpp from η_integral
START = time.time()
exact.CqqCpp_η(params, ω, title + '_η', η = η, J = J, ωmax = ωmax)
END = time.time()
print(f'CqqCpp computation time = {END-START} seconds')
print()
exit()

### CqqCpp from Gp ('data.py' should include the function 'ηt')
# tf = 100; dt = 1e-2; t = np.linspace(0., tf, int(tf/dt)+1)  # time grid
# ηt = data.ηt(t)
# Gp = exact.compute_Gp(params, t, ηt, title)
# exact.CqqCpp_Gp(params, t, Gp, ω, title + '_Gp')
