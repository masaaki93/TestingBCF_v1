import numpy as np
import importlib
# import mpmath
import input.expfit as expfit
import os, re
from settings import SD

### spectral density name
SD_name = SD()
q2_eq = 0.3819117526320983
p2_eq = 1.1771511040824276

### input data (reorganization energy & title)
module_name = f'{SD_name}.data'
data = importlib.import_module(module_name)
dir_name = f'{SD_name}/{data.get_title()}'

Lt_filename = f'{dir_name}/data_time.csv'
Lt_data = np.loadtxt(Lt_filename, dtype=complex)
t = Lt_data[:,0].real
Lt = Lt_data[:,1]

Lω_filename = f'{dir_name}/data_frequency.csv'
Lω_data = np.loadtxt(Lω_filename)
ω = Lω_data[:,0]
Lω = Lω_data[:,1]

exact_fourier = np.loadtxt(f'{dir_name}/exact_η_corr_fourier.csv', skiprows=1)

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

### main

f = open(f'{dir_name}/{algorithm_dir_name}/_error.csv', mode="w")
# f.write('  K  :  error_L_time (integral)  :  error_L_frequency (integral)  :  error_L_time (data)  :  error_L_frequency (data)  :  q2  :  p2  :  δq2  :  δp2  :  δCqqω (data)  :  δCppω (data)  :' + '\n')
f.write('  K  :  error_L_time (data)  :  error_L_frequency (data)  :  q2  :  p2  :  error_q2  :  error_p2  :  error_Cqq_frequency (data)  :  error_Cpp_frequency (data)  :' + '\n')
for i in range(len(results_name)):

    # input results
    dir_name_heom = f'{dir_name}/{results_name[i]}'
    K, d, z, η = expfit.read_correlation_function(dir_name_heom)
    obs = np.loadtxt(f'{dir_name_heom}/heom_obs.csv', skiprows = 1)

    # δLt_integral = 1/(tf-ti) \int_ti^tf |L(t)-L_{mod}(t)| dt
    # ti = 0; tf = 10
    # δLt_integrand = lambda t: abs( data.Lt_exact(float(t)) - expfit.model_Lt(float(t), d, z) )
    # δLt_integral = 1/(tf-ti) * float(mpmath.quad(δLt_integrand, [ti, tf]))

    # δLω_integral = 1/(ωf-ωi) \int_ωi^ωf |F[L](ω)-F[L_{mod}](ω)| dω
    # ωf = 1e+2; ωi = - ωf
    # δLω_integrand = lambda ω: abs( data.Lω_exact(float(ω)) - expfit.model_Lω(float(ω), d, z, η) )
    # δLω_integral = 1/(ωf-ωi) * float(mpmath.quad(δLω_integrand, [ωi, ωf]))

    #  δLt_data = (1/N) * \sum_{i=0}^{N-1} |L(t[i])-L_{mod}(t[i])| (absolute error)
    Lt_model = expfit.model_Lt(t, d, z)
    δLt_data = 1/len(t) * np.sum(np.abs(Lt_model - Lt))

    #  δLω_data = (1/N) * \sum_{i=0}^{N-1} |F[L](ω[i])-F[L_{mod}](ω[i])| (absolute error)
    Lω_model = expfit.model_Lω(ω, d, z, η)
    δLω_data = 1/len(ω) * np.sum( np.abs(Lω_model - Lω))

    # q2, p2 (relative error)
    q2 = obs[-1,3]
    p2 = obs[-1,4]
    δq2 = np.abs(q2-q2_eq)/q2_eq
    δp2 = np.abs(p2-p2_eq)/p2_eq

    # δC_data (absolute error)
    fourier = np.loadtxt(f'{dir_name_heom}/heom_corr_fourier.csv', skiprows = 1)
    δCqqω_data = np.sum( np.abs(exact_fourier[:,1] - fourier[:,1]) / len(fourier[:,1]) )
    δCppω_data = np.sum( np.abs(exact_fourier[:,2] - fourier[:,2]) / len(fourier[:,2]) )

    f.write('{:3.0f}'.format(K))
    # f.write('{:20.5e}'.format(δLt_integral))
    # f.write('{:20.5e}'.format(δLω_integral))
    f.write('{:20.5e}'.format(δLt_data))
    f.write('{:20.5e}'.format(δLω_data))
    f.write('{:24.8e}'.format(q2))
    f.write('{:24.8e}'.format(p2))
    f.write('{:24.8e}'.format(δq2))
    f.write('{:24.8e}'.format(δp2))
    f.write('{:20.5e}'.format(δCqqω_data))
    f.write('{:20.5e}'.format(δCppω_data))
    f.write('\n')
