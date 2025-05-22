# Bath Correlation Function Tests Using a Harmonic Oscillator System

This Python package enables testing of model bath correlation functions (BCF) for open quantum dynamics simulations. In particular, it allows benchmarking the accuracy of thermalization behavior against exact solutions using a harmonic oscillator system. The methodology is detailed in:

Masaaki Tokieda, *Testing bath correlation functions for open quantum dynamics simulations*, [arXiv:2504.08068 [quant-ph]](https://doi.org/10.48550/arXiv.2504.08068)

---

## Usage

The testing procedure consists of the following six steps:

1. **Prepare the spectral density information**
2. **Set system parameters and spectral density** (`system.py`)
3. **Compute exact solutions** (`exact.py`)
4. **Construct model BCFs** (`fitting.py`)
5. **Solve the HEOM** (`heom.py`)
6. **Evaluate the error** (`error.py`)

---

### Spectral Density Information

The package includes several common spectral densities, such as:

**brownian**  
```math
J(ω) = ξ γ² ωb² ω / ((ω² − ωb²)² + (γ ω)²)
```

**drude**  
```math
J(ω) = ξ γ² ω / (γ² + ω²)
```

**expcutoff**  
```math
J(ω≧0) = (π/2) α ωc (ω/ωc)ˢ exp(−ω/ωc)
```

**circcutoff**  
```math
J(ω≧0) = (π/2) α ωc (ω/ωc)ˢ [1 − (ω/ωc)²]ᵐ  θ(ωc−ω)
```

**purcell** is an example of numerical data taken from the GitHub repository [`spectral_density_fit`](https://github.com/jfeist/spectral_density_fit)

Parameters for the spectral densities are defined in the `data.py` file located in each directory.

---

### Creating a Custom `data.py`

If your target spectral density is not already included, you can create a new `data.py` file. This file should define the following required functions:

```python
def get_title():          # Returns a string describing the parameter set
def get_λ():              # Reorganization energy (used in heom.py)
def get_β():              # Inverse temperature (used in exact.py); set β = -1 for zero temperature
def get_η():              # Laplace transform of the friction kernel η̂(s) for Re(s) ≥ 0 (used in exact.py); set η == None if not available
def output_Lt():          # Exports the BCF data in the time domain (used in fitting.py)
def output_Lω():          # Exports the BCF data in the frequency domain (used in fitting.py)
```

**Optional functions:**

```python
def J_(ω):                # J(ω) for ω ≥ 0 (used to numerically compute the exact solutions in exact.py if η is not available)
def J(ω):                 # J(ω) over ℝ (used for plotting in exact.py)
def Lt_exact(t):          # Exact BCF in the time domain (used in error.py)
def Lω_exact(ω): 　　　　　 # Exact BCF in the frequency domain (used in error.py)
```

---

## Main Files Overview

### `system.py`

Defines the system (oscillator) parameters and the directory name containing the spectral density data.

---

### `exact.py`

Computes exact equilibrium expectation values and correlation functions.

- The analytic formula for equilibrium expectation values includes an infinite sum—adjust the `tol` parameter to check convergence.
- If η is not available, the Laplace transform η̂(s) is computed numerically using an integral (see Eqs. (10) and (11) in the paper). In this case, you must set the upper bound for the ω-integral (`ωmax`).

---

### `fitting.py`

Fits the BCF in the time and frequency domains using a linear combination of complex exponentials.

- Input data is generated via `output_Lt()` and `output_Lω()` in `data.py`.
- See Section IV.A of the paper for details about the methods
- Fitting results are saved as shown [here](Lmod.pdf)

---

### `heom.py`

Solves the HEOM in the moment representation using the fitted model BCFs.

- Specify the fitting algorithm:
  - `algorithm_name`: AAA, ESPRIT, IP, or GMT
  - `algorithm_dir_name`: name of the directory containing the fitting results
- Tune the following parameters for convergence checks:
  - `tss`: time for steady-state computation
  - `tf`: maximum time for correlation function computation
  - `dt_corr`: time step for Fourier transform

---

### `error.py`

Evaluates the errors in the model BCF, the equilibrium expectation values, and the autocorrelation functions.

- Specify the equilibrium expectation values `q2_eq` and `p2_eq` obtained from `exact.py`

---

## Environment

This package has been tested with the following software versions:

- Python **3.13.2**
- NumPy **2.2.6**
- SciPy **1.15.3**
- Matplotlib **3.10.3**
- QuTiP **5.1.1**
- mpmath **1.3.0** (required for `expcutoff` and BCF error evaluation by integral in `error.py`)
- [`spectral_density_fit`](https://github.com/jfeist/spectral_density_fit) **0.2.1** (required for using the method IP in `fitting.py`)
