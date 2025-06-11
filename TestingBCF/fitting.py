import numpy as np
import importlib
import os
from system import SD

import input.expfit as expfit # For AAA and ESPRIT
# import input.expfit_IP as expfit_IP # For IP. Install 'spectral_density_fit'.
# import input.expfit_GMT as expfit_GMT # For GMT&FIT

### SD_name
SD_name = SD()

### input data
module_name = f'{SD_name}.data'
data = importlib.import_module(module_name)

### make directory if not exist
dir_name = f'{SD_name}/{data.get_title()}'
os.makedirs(dir_name, exist_ok=True)

##########################################
# obtain and plot data
##########################################
data.output_Lt()
data.output_Lω()

##########################################
# Fitting
##########################################

Lt_filename = f'{dir_name}/data_time.csv'
Lω_filename = f'{dir_name}/data_frequency.csv'

Lω_data = np.loadtxt(Lω_filename)
ω = Lω_data[:,0]

"""
    AAA
"""
# Klbs = [i for i in range(2,21,2)]
# for i in range(len(Klbs)):
#     _, _, _ = expfit.AAA_K_fit(Lω_filename, Klbs[i], Lt_filename = Lt_filename, dir = f'{dir_name}/AAA')
#     print(f'AAA{Klbs[i]} DONE')
# exit()


"""
    ESPRIT
"""
Ms = [i for i in range(1,11)]
for i in range(len(Ms)):
    expfit.ESPRIT_fit(Lt_filename, Ms[i], Lω_filename = Lω_filename, dir = f'{dir_name}/ESPRIT')
    print(f'ESPRIT{Ms[i]} DONE')
exit()


"""
    IP
"""
# opttol_rel = 1e-4
#
# Klbs = [i for i in range(2,21,2)]
# for i in range(len(Klbs)):
#     d, z, _ = expfit.AAA_K_fit(Lω_filename, Klbs[i], Lt_filename = Lt_filename, dir = f'{dir_name}/_')
#     H_init, κ_init, g_init = expfit_IP.IP_initial_guess_naive(d, z)
#     expfit_IP.IP_fit(Lω_filename, H_init, κ_init, g_init, fitlog = False, Lt_filename = Lt_filename, opttol_rel = opttol_rel, dir = f'{dir_name}/IP{opttol_rel:.0e}')
#     print(f'IP{Klbs[i]} DONE')
# exit()


"""
    GMT&FIT
"""
# ### Fitting spectral density
# Mj = 10
# c, μ = expfit_GMT.cμ_ESPRIT_fit(Lt_filename, Mj)
# β = data.get_β()    # β = -1 (zero temperature) is not implemented yet
#
# dir_name_GMT = f'{dir_name}/GMT_Mj{Mj}_ReLt'
# os.makedirs(dir_name_GMT, exist_ok=True)
#
# # print('GMT&FIT: Fitting the spectral density')
# # print(f'c={c}'); print(f'μ={μ}'); print()
#
# # ωJ = np.linspace(0, 10, 100)
# ωJ = np.arange(0, 100, 1e-3)
# expfit_GMT.plot_J(ωJ, data.J(ωJ), c, μ, f'{dir_name_GMT}/Jfit')
#
# ### Fitting Matsubara modes
# K_MTs = [1,2,3,4,5]
# LT = 1
#
# for j in range(len(K_MTs)):
#     K_MT = K_MTs[j]
#     if K_MT == 0:
#         η = 0.
#         d, z = expfit_GMT.cμ_to_aJzJ(c, μ, β)
#     else:
#         # d, z, η = expfit_GMT.MT_FD_fit(c, μ, β, K_MT, ω, LT = LT, opttype = 'integral')
#         # d, z, η = expfit_GMT.MT_FD_fit(c, μ, β, K_MT, ω, LT = LT, opttype = 'points')
#         d, z, η = expfit_GMT.MT_ReLt_fit(Lt_filename, c, μ, β, K_MT)
#         # d, z, η = expfit_GMT.MT_Pade(c, μ, β, K_MT)    # 1 ≦ K_MT ≦ 5
# 
#     d, z, K = expfit.regularization(d, z, 1e-10)
#     expfit.savefile(Lt_filename, Lω_filename, d, z, η, K, f'{dir_name_GMT}/GMT_K{K}')
