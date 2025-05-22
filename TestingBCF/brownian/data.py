import numpy as np
import matplotlib.pyplot as plt

# bath parameters; J(ω) = ξ γ^2 ωb^2 ω / ((ω^2 - ωb^2)^2 + (γ ω)^2)
SD_name = 'brownian'

γ = 1    # cutoff frequency
ωb = 1
ξ = 1; λ = ξ * γ / 2
β = 1     # inverse temperature (must be finite)

Number_of_Matsubara_terms = 100000

#
# get
#

def get_λ():
    return λ

def get_β():
    return β

def get_title():
    return f'_β{β}_ωb{ωb}_γ{γ}_ξ{ξ}'

#
# spectral density / correlation function
#

def J_(ω):   # ω >= 0 (faster)
    return ξ * γ**2 * ωb**2 * ω / ((ω**2 - ωb**2)**2 + (γ * ω)**2)

def J(ω):   # ω can be negative too
    return J_(ω)

def Lω_exact(ω):
    result = np.where(
        ω == 0,
        2 * ξ * γ**2 / (β * ωb**2),  # This handles the case when ω == 0
        2 * J(ω) / (1 - np.exp(- β * ω))  # Otherwise
    )
    return result

def Lt_exact(t):
    L = 0
    if ωb != (γ/2):
        if ωb > (γ/2):
            ωb_tilde = np.sqrt(ωb**2 - (γ/2)**2)
        elif ωb < (γ/2):
            ωb_tilde = 1j * np.sqrt((γ/2)**2 - ωb**2)

        a_spec = np.zeros(2, dtype=complex)
        a_spec[0] = ξ*γ*ωb**2 / (4*ωb_tilde) * (1/np.tanh(β/2 * (ωb_tilde + 1j*γ/2)) - 1)    # exp(- ((γ/2) - 1j ωb_tilde)t)
        a_spec[1] = ξ*γ*ωb**2 / (4*ωb_tilde) * (1/np.tanh(β/2 * (ωb_tilde - 1j*γ/2)) + 1)    # exp(- ((γ/2) + 1j ωb_tilde)t)

        γ_spec = np.zeros_like(a_spec)
        γ_spec[0] = (γ/2) - 1j * ωb_tilde    # exp(- ((γ/2) - 1j ωb_tilde)t)
        γ_spec[1] = (γ/2) + 1j * ωb_tilde    # exp(- ((γ/2) + 1j ωb_tilde)t)

        for k in range(2):
            L += a_spec[k] * np.exp(- γ_spec[k] * t)

    elif ωb == (γ/2):
        L = (ξ*γ*ωb**2*β / 4 * (2*t/β/np.tan(β*γ/4) + 1/np.sin(β*γ/4)**2) - 1j * ξ*γ*ωb**2 * t / 2) * np.exp(- γ*t/2)

    for k in range(Number_of_Matsubara_terms):
        ν_k = 2 * k * np.pi / β
        L -= (2*ξ*γ**2*ωb**2/β) * ν_k / ((ν_k**2 + ωb**2)**2 - (γ*ν_k)**2) * np.exp(- ν_k * t)
    return L

def ηt(t): # ηt: friction kernel in the time domain. Evaluated analytically
    return ξ * γ * np.exp(- γ * t)

    if ωb != (γ/2):
        if ωb > (γ/2):
            ωb_tilde = np.sqrt(ωb**2 - (γ/2)**2)
        elif ωb < (γ/2):
            ωb_tilde = 1j * np.sqrt((γ/2)**2 - ωb**2)

        return ξ*γ * (np.cos(ωb_tilde*t) + γ/(2*ωb_tilde)*np.sin(ωb_tilde*t)) * np.exp(- γ*t/2)

    elif ωb == (γ/2):
        return ξ*γ * (1 + γ*t/2) * np.exp(- γ*t/2)


def get_η(): # η: friction kernel in the Laplace domain

    print('Analytic hat{η}(z) is used')
    print()

    if ωb != (γ/2):
        if ωb > (γ/2):
            ωb_tilde = np.sqrt(ωb**2 - (γ/2)**2)
        elif ωb < (γ/2):
            ωb_tilde = 1j * np.sqrt((γ/2)**2 - ωb**2)

        def η(z):
            return ξ*γ / 2 * ((1+1j*γ/(2*ωb_tilde))/(z+γ/2+1j*ωb_tilde) + (1-1j*γ/(2*ωb_tilde))/(z+γ/2-1j*ωb_tilde))

    elif ωb == (γ/2):
        def η(z):
            return ξ*γ / (z+γ/2) + ξ*γ**2/2 / (z+γ/2)**2

    return η

#
# plot
#

def output_Lt():

    filename = f'{SD_name}/{get_title()}/data_time'

    # y = γ * t
    y_max = 50
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
    x_min = -10
    x_max = +40
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
