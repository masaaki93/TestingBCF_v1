import numpy as np
from scipy import sparse
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from input.expfit import read_correlation_function

class HEOMSolver:
    def __init__(self, dir_name_heom, ω0, v0, λ, tss, dt_te, dt_corr, tf, ω):

        K, d, z, η = read_correlation_function(dir_name_heom)

        self.title = dir_name_heom + '/heom'
        self.ω0 = ω0
        self.v0 = v0
        self.λ = λ
        self.η = η
        self.d = d
        self.z = z
        self.ω = ω
        self.tss = tss
        self.dt_te = dt_te
        self.dt_corr = dt_corr
        self.tf = tf
        self.K = K

        self.hierarchy = self.set_hierarchy(K + 2, Hmax=2)
        self.dim_Liouv = len(self.hierarchy)

        self.q_l = None
        self.p_l = None
        self.Liouv = None

    def write_settings(self):
        with open(self.title + '_settings.csv', mode='w') as f:
            f.write('*** Target system: oscillator system\n')
            f.write(f'ω0 = {self.ω0}\n')
            f.write(f'v0 = {self.v0}\n\n')

            f.write('*** Correlation function: L(t>=0) = \sum_{k = 0}^{K-1} d[k] exp(- z[k] t) + 2 \eta \delta(t)\n')
            f.write(f'K: {len(self.d)}\n')

            for k in range(len(self.d)):
                d_k = self.d[k]
                z_k = self.z[k]
                f.write(f'd[{k}] = {d_k.real:8.3e} + {d_k.imag:8.3e} i     |     '
                        f'z[{k}] = {z_k.real:8.3e} + {z_k.imag:8.3e} i\n')

            f.write(f'η = {self.η:8.3e}\n\n')

            f.write('*** Integrator: applying expm(Liouv * t)\n')
            f.write(f'Steady-state computation: tss = {self.tss} (dt_te = {self.dt_te})\n')
            f.write(f'Correlation function: tf = {self.tf} (dt_corr = {self.dt_corr})\n')

    def set_order_H(self, N, H, partial=[]):
        if N == 1:
            return [partial + [H]]
        lst = []
        for i in range(H + 1):
            lst.extend(self.set_order_H(N - 1, H - i, partial + [i]))
        return lst

    def set_hierarchy(self, N, Hmax):
        hierarchy = []
        for H in range(Hmax + 1):
            hierarchy.extend(self.set_order_H(N, H))
        return hierarchy

    def operators(self):

        def dagger(A):
            return A.conj().T

        def dbar(k):
            kbar = np.where(self.z == np.conjugate(self.z[k]))[0]
            if len(kbar) == 1:
                return np.conjugate(self.d[kbar[0]])
            elif len(kbar) == 0:
                raise ValueError('z not closed under complex conjugation.')
            else:
                raise ValueError('Duplicated conjugates in z.')

        def ladder(k):
            B_k = sparse.lil_matrix((self.dim_Liouv, self.dim_Liouv), dtype=np.float64)
            for index, list_index in enumerate(self.hierarchy):
                list_new = list(list_index)
                list_new[k] += 1
                if list_new in self.hierarchy:
                    index_new = self.hierarchy.index(list_new)
                    B_k[index, index_new] = np.sqrt(list_index[k] + 1)
            return B_k.tocsr()

        # l: left
        # r: right
        # c: commutator
        # a: anti-commutator

        B_0 = ladder(0)
        B_1 = ladder(1)

        N_c = dagger(B_0) @ B_0 - dagger(B_1) @ B_1
        q_l = (B_0 + dagger(B_0)) / np.sqrt(2) + B_1 / np.sqrt(2)
        q_r = (B_1 + dagger(B_1)) / np.sqrt(2) + B_0 / np.sqrt(2)
        q_c = q_l - q_r
        q_a = q_l + q_r

        p_l = 1j * (dagger(B_0) - B_0) / np.sqrt(2) + 1j * B_1 / np.sqrt(2)

        Liouv = - (1j * self.ω0 * N_c + 1j * self.λ * self.v0**2 * q_c @ q_a + self.η * self.v0**2 * q_c @ q_c)

        for k in range(len(self.d)):
            B_k = ladder(k + 2)
            Liouv -= self.z[k] * dagger(B_k) @ B_k
            Liouv -= self.v0 * q_c @ B_k
            Liouv += self.v0 * dagger(B_k) @ (self.d[k] * q_l - dbar(k) * q_r)

        self.q_l = sparse.csr_matrix(q_l)
        self.p_l = sparse.csr_matrix(p_l)
        self.Liouv = sparse.csr_matrix(Liouv)

    def time_evolution(self):

        print("=== time evolution ===")

        t = np.arange(0., self.tf + self.dt_te, self.dt_te)
        obs_exp = np.zeros((5, len(t)))
        y = np.zeros(self.dim_Liouv, dtype=np.complex128)
        y[0] = 1.

        def comp_obs_exp(f, y, i):
            obs_exp[0, i] = (self.q_l @ y)[0].real
            obs_exp[1, i] = (self.p_l @ y)[0].real
            obs_exp[2, i] = (self.q_l @ self.q_l @ y)[0].real
            obs_exp[3, i] = (self.p_l @ self.p_l @ y)[0].real
            obs_exp[4, i] = (0.5 * ((self.q_l @ self.q_l + self.p_l @ self.p_l) @ y - y))[0].real
            f.write(f'{t[i]:5.3f}')
            for k in range(obs_exp.shape[0]):
                f.write(f'{obs_exp[k,i]:18.8e}')
            f.write('\n')

        with open(self.title + '_obs.csv', 'w') as f_obs:
            f_obs.write('t  :   <q>   :   <p>   :   <q^2>   :   <p^2>    :   <N>    \n')
            comp_obs_exp(f_obs, y, 0)
            exp_Lt = sparse.linalg.expm(self.Liouv * self.dt_te)
            for i in range(len(t) - 1):
                y = exp_Lt @ y
                comp_obs_exp(f_obs, y, i + 1)
                print(f't = {t[i]:.2f} / {self.tf}:  tr(rho_S) = {y[0]}')

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = [r'$\langle q \rangle$', r'$\langle p \rangle$', r'$\langle q^2 \rangle$', r'$\langle p^2 \rangle$', r'$\langle N \rangle$']
        for k in range(5):
            ax.plot(t, obs_exp[k], label=labels[k], lw=2)
        ax.set_xlabel(r'$t$', fontsize=15)
        ax.set_ylabel('Expectation values', fontsize=15)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(self.title + '_obs.png')
        plt.close()

        return obs_exp[:, -1], y

    def equilibrium_correlation_function(self):
        obs_exp_ss, y = self.time_evolution()

        print(f'<q>_eq = {obs_exp_ss[0]}')
        print(f'<p>_eq = {obs_exp_ss[1]}')
        print(f'<q^2>_eq = {obs_exp_ss[2]}')
        print(f'<p^2>_eq = {obs_exp_ss[3]}')
        print(f'<N>_eq = {obs_exp_ss[4]}')
        print()

        print("=== correlation function ===")

        t = np.arange(0., self.tf + self.dt_corr, self.dt_corr)
        corr = np.zeros((2, len(t)), dtype=np.complex128)
        corr_offset = np.array([obs_exp_ss[0] ** 2, obs_exp_ss[1] ** 2])
        y_q = self.q_l @ y
        y_p = self.p_l @ y

        def corr_expectation(y_q, y_p, i):
            corr[0, i] = (self.q_l @ y_q)[0] - corr_offset[0]
            corr[1, i] = (self.p_l @ y_p)[0] - corr_offset[1]

        corr_expectation(y_q, y_p, 0)
        exp_Lt = sparse.linalg.expm(self.Liouv * self.dt_corr)
        for i in range(len(t) - 1):
            y_q = exp_Lt @ y_q
            y_p = exp_Lt @ y_p
            corr_expectation(y_q, y_p, i + 1)
            print(f't = {t[i]:.2f} / {self.tf}')

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(t, corr[0].real, label=r'${\rm Re}C_{qq}(t)$', lw=2)
        ax.plot(t, corr[1].real, label=r'${\rm Re}C_{pp}(t)$', lw=2)
        ax.plot(t, corr[0].imag, '--', label=r'${\rm Im}C_{qq}(t)$', lw=2)
        ax.plot(t, corr[1].imag, '--', label=r'${\rm Im}C_{pp}(t)$', lw=2)
        ax.set_xlabel(r'$t$', fontsize=15)
        ax.set_ylabel('Correlation function', fontsize=15)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(self.title + '_corr_time.png')
        plt.close()

        corr_fourier = np.zeros((2, len(self.ω)))
        with open(self.title + '_corr_fourier.csv', 'w') as f:
            f.write('omega  :   F[Cqq]   :   F[Cpp]\n')
            for j, omega in enumerate(self.ω):
                f.write(f'{omega:12.3e}')
                for k in range(2):
                    val = 2 * simpson(y=corr[k] * np.exp(1j * omega * t), x=t).real
                    corr_fourier[k, j] = val
                    f.write(f'{val:20.8e}')
                f.write('\n')
