import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.constants import c
from astropy import units as u

file_path = "/Pantheon+SH0ES (1).dat"

data = pd.read_csv(file_path, delim_whitespace=True, comment="#")

print(data.head())

print(data.columns)

clean_data = data.dropna(subset=['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG'])

z = clean_data['zHD'].values
mu_obs = clean_data['MU_SH0ES'].values
mu_err = clean_data['MU_SH0ES_ERR_DIAG'].values

print("Cleaned redshift (z):", z[:5])
print("Cleaned distance modulus (mu_obs):", mu_obs[:5])
print("Cleaned uncertainty (mu_err):", mu_err[:5])

plt.figure(figsize=(8, 6))

plt.errorbar(z, mu_obs, yerr=mu_err, fmt='o', color='black', ecolor='gray',
             elinewidth=0.8, capsize=2, markersize=3, label='Observed SNe Ia')

plt.xscale('log')

plt.xlabel("Redshift (z)")
plt.ylabel("Distance Modulus (μ)")
plt.title("Hubble Diagram (Pantheon+SH0ES)")

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()



c_kms = c.to('km/s').value  # = 299792.458

def E(z, Omega_m):
    return np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

def luminosity_distance(z, H0, Omega_m):
    integral, _ = quad(lambda z_prime: 1 / E(z_prime, Omega_m), 0, z)
    return (c_kms / H0) * (1 + z) * integral

def mu_theory(z_array, H0, Omega_m):
    
    return np.array([5 * np.log10(luminosity_distance(z, H0, Omega_m)) + 25 for z in z_array])


from scipy.optimize import curve_fit

p0 = [70, 0.3]

params, covariance = curve_fit(mu_theory, z, mu_obs, sigma=mu_err, p0=p0, absolute_sigma=True)

H0_fit, Omega_m_fit = params
H0_err, Omega_m_err = np.sqrt(np.diag(covariance))

print(f"Fitted H0 = {H0_fit:.2f} ± {H0_err:.2f} km/s/Mpc")
print(f"Fitted Omega_m = {Omega_m_fit:.3f} ± {Omega_m_err:.3f}")

from scipy.integrate import quad

def age_of_universe(H0, Omega_m):
    H0_SI = H0 * (1000) / (3.085677581491367e22)  # km/s/Mpc → m/s/m → 1/s

    def integrand(z):
        return 1 / ((1 + z) * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m)))

    integral, _ = quad(integrand, 0, np.inf)

    age_seconds = integral / H0_SI

    age_gyr = age_seconds / (60 * 60 * 24 * 365.25 * 1e9)

    return age_gyr

t0 = age_of_universe(H0_fit, Omega_m_fit)
print(f"Estimated age of Universe: {t0:.2f} Gyr")

mu_model = mu_theory(z, H0_fit, Omega_m_fit)

residuals = mu_obs - mu_model

plt.figure(figsize=(8, 5))
plt.scatter(z, residuals, s=8, color='darkred', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.xscale('log')

plt.xlabel("Redshift (z)")
plt.ylabel("Residual (μ_obs - μ_model)")
plt.title("Residuals of Hubble Diagram Fit")
plt.grid(True, which='both', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

def mu_fixed_Om(z_array, H0):
    return mu_theory(z_array, H0, Omega_m=0.3)
params_fixed, cov_fixed = curve_fit(mu_fixed_Om, z, mu_obs, sigma=mu_err, p0=[70], absolute_sigma=True)

H0_fixed = params_fixed[0]
H0_fixed_err = np.sqrt(np.diag(cov_fixed))[0]

print(f"Fitted H0 with Omega_m fixed at 0.3 = {H0_fixed:.2f} ± {H0_fixed_err:.2f} km/s/Mpc")

z_split = 0.1

z_low = z[z < z_split]
mu_low = mu_obs[z < z_split]
mu_err_low = mu_err[z < z_split]

z_high = z[z >= z_split]
mu_high = mu_obs[z >= z_split]
mu_err_high = mu_err[z >= z_split]

H0_low, cov_low = curve_fit(mu_fixed_Om, z_low, mu_low, sigma=mu_err_low, p0=[70], absolute_sigma=True)

H0_high, cov_high = curve_fit(mu_fixed_Om, z_high, mu_high, sigma=mu_err_high, p0=[70], absolute_sigma=True)

print(f"Low-z (z < {z_split}): H₀ = {H0_low[0]:.2f} km/s/Mpc")
print(f"High-z (z ≥ {z_split}): H₀ = {H0_high[0]:.2f} km/s/Mpc")

