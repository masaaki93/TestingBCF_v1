import numpy as np
import matplotlib.pyplot as plt

# bath parameters; J(ω) = 2 λ γ ω / (γ^2 + ω^2) = ξ γ^2 ω / (γ^2 + ω^2) (λ = ξ γ / 2)
SD_name = 'drude'

γ = 1    # cutoff frequency
ξ = 1; λ = ξ * γ / 2
# λ = 0.1; ξ = 2 * λ / γ
β = 10     # inverse temperature (must be finite)
α = β * γ / 2     # α = β hbar γ / 2

Number_of_Matsubara_terms = 1000000

#
# get
#

def get_λ():
    return λ

def get_β():
    return β

def get_γ():
    return γ

def get_title():
    return f'_β{β}_γ{γ}_ξ{ξ}'

#
# spectral density / correlation function
#

def J_(ω):   # ω >= 0 (faster)
    return 2 * λ * γ * ω / (ω ** 2 + γ ** 2)

def J(ω):   # ω can be negative too
    return J_(ω)

def Lω_exact(ω):
    result = np.where(
        ω == 0,
        2 * λ / α,  # This handles the case when ω == 0
        2 * J(ω) / (1 - np.exp(- β * ω))  # Otherwise
    )
    return result

def Lt_exact(t):
    L = λ * γ * (1/np.tan(α) - 1j) * np.exp(- γ * t)
    for k in range(Number_of_Matsubara_terms):
        nu_k = 2 * k * np.pi / β
        L -= (4 * λ * γ / β) * nu_k / (γ ** 2 - nu_k ** 2) * np.exp(- nu_k * t)
    return L

def ηt(t): # ηt: friction kernel in the time domain. Evaluated analytically
    return ξ * γ * np.exp(- γ * t)

def get_η(): # η: friction kernel in the Laplace domain

    print('Analytic hat{η}(z) is used')
    print()

    def η(z):
        return ξ * γ / (z + γ)

    return η

#
# plot
#

def output_Lt():

    filename = f'{SD_name}/{get_title()}/data_time'

    # y = γ * t
    y_max = 100
    dy = 1e-1
    y = np.arange(0., y_max, dy)
    t = y/γ
    Lt = Lt_exact(t)
    f = open(filename + '.csv', mode="w")
    for l in range(len(t)):
        f.write('{:12.5e}'.format(t[l]) + '{:50.15e}'.format(Lt[l]) + '\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(t,Lt.real,color='red',lw=3,label='Real')
    plt.plot(t,Lt.imag,color='blue',lw=3,label='Imag')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$t$", fontsize=15)
    plt.ylabel(r"$L(t)$", fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(filename + '.png', format='png')
    plt.close()

def output_Lω():

    filename = f'{SD_name}/{get_title()}/data_frequency'

    # x = ω / γ
    x_min = -20
    x_max = +80
    dx = 5e-2
    x = np.arange(x_min, x_max, dx)
    ω = γ*x
    Lω = Lω_exact(ω)
    f = open(filename + '.csv', mode="w")
    for l in range(len(ω)):
        f.write('{:12.5e}'.format(ω[l]) + '{:25.15e}'.format(Lω[l]) + '\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ω,Lω,color='red',lw=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=15)
    plt.ylabel(r"$\mathcal{F}[L](\omega)$", fontsize=15)
    plt.tight_layout()
    plt.savefig(filename + '.png', format='png')
    plt.close()
