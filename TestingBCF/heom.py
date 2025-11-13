import numpy as np
import importlib
import os, re
from settings import SD, ω0v0

import time

from input.heom import HEOMSolver
import matplotlib.pyplot as plt

### system parameters
ω0, v0 = ω0v0()

### spectral density name
SD_name = SD()

### input data (reorganization energy & title)
module_name = f'{SD_name}.data'
data = importlib.import_module(module_name)
λ = data.get_λ()
dir_name = f'{SD_name}/{data.get_title()}'

### results name
algorithm_name = 'ESPRIT'
algorithm_dir_name = algorithm_name
# Ks = [i for i in range(2,11,2)]
# results_name = [f'{algorithm_dir_name}/{algorithm_name}_K{K}' for K in Ks]

base_names = set()
for filename in os.listdir(f'{dir_name}/{algorithm_dir_name}'):
    match = re.match('(' + algorithm_name + r'_K\d+)', filename)
    if match:
        base_names.add(match.group(1))
filename = sorted(base_names, key=lambda x: int(re.search(r'\d+', x).group()))
results_name = [f'{algorithm_dir_name}/{filename[i]}' for i in range(len(filename))]

### Time
tss = 30    # Max time for steady state computation
dt_te = tss/200      # Time step for computing the time evolution
tf = tss  # Max time for computing the correlation function
dt_corr = 1e-1      # Time step for computing the correlation function

### Comparison of F[Cqq], F[Cpp] with the exact solutions
CqqCpp_exact = True   # True = yes, we compare / False = no, we don't compare

if CqqCpp_exact:
    exact = np.loadtxt(f'{dir_name}/exact_η_corr_fourier.csv', skiprows=1)
    ω = exact[:,0]
else:
    ω = np.linspace(-2,10,1000)

##########################################
# Solvign HEOM in the moment representation
##########################################

for i in range(len(results_name)):

    dir_name_heom = f'{dir_name}/{results_name[i]}'
    os.makedirs(dir_name_heom, exist_ok=True)

    solver = HEOMSolver(dir_name_heom=dir_name_heom,
                        ω0=ω0, v0=v0, λ=λ,
                        tss=tss, dt_te=dt_te, dt_corr=dt_corr, tf=tf,
                        ω=ω)

    solver.operators()
    solver.write_settings()

    # START = time.time()
    # _, _ = solver.time_evolution()    # 0 <= t <= tf
    # END = time.time()
    # print(f'Elapsed time: {END-START} seconds')

    START = time.time()
    solver.equilibrium_correlation_function()
    END = time.time()
    print(f'Elapsed time: {END-START} seconds')

    if CqqCpp_exact:
        heom = np.loadtxt(f'{solver.title}_corr_fourier.csv', skiprows=1)

        Cωmin = np.min( np.hstack([exact[:,1], exact[:,2]])   )
        Cωmax = np.max( np.hstack([exact[:,1], exact[:,2]])   )
        # Cωmax = np.max( exact[:,2] )

        ΔCω = (Cωmax-Cωmin) / 10

        # overall behavior
        plt.figure(figsize=(7,5))
        plt.plot(exact[:,0], exact[:,1], label=r'$\mathcal{F}[C_{qq}](\omega)$', color='red', lw='1.5')
        plt.plot(heom[:,0], heom[:,1], label=r'$\mathcal{F}[C_{qq}^{\rm HEOM}](\omega)$', linestyle = 'None', color='red', marker='o', markersize=4)
        plt.plot(exact[:,0], exact[:,2], label=r'$\mathcal{F}[C_{pp}](\omega)$', color='blue', lw='1.5')
        plt.plot(heom[:,0], heom[:,2], label=r'$\mathcal{F}[C_{pp}^{\rm HEOM}](\omega)$', linestyle = 'None', color='blue', marker='o', markersize=4)
        plt.ylim(Cωmin-ΔCω,Cωmax+ΔCω)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.xlabel(r'$\omega$', fontsize=18)
        plt.ylabel('Correlation function', fontsize=18)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{solver.title}_corr_fourier.png', format='png')
        plt.close()  # Close the first figure
