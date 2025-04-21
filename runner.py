import numpy as np
from scipy.constants import speed_of_light, nu2lambda, lambda2nu
import matplotlib.pyplot as plt

import pulse
from utils import fwhm2std, calc_pdf_std

# Global parameters:
wavelength = 1.55e-6 # meters;
freq0 = lambda2nu(wavelength) # Hertz;
w0 = 2*np.pi*freq0
pulse_FWHM = 20e-12  # seconds;
pulse_std = fwhm2std(pulse_FWHM)
alpha = 0.0
beta = np.array([6.0e6, 5.0e3, -2.0e-2]) * np.array([1.0, 1e-12, 1e-24]) # rad/m; s/m; s^2/m.
ng = speed_of_light*beta[1]

# Analytical FWHM pulse spread after propagation:
zi = 10e0 # meters.
zf = 1000e3 # meters.
num_zz_points = int(1e2)
zz = np.logspace(np.log10(zi), np.log10(zf), num_zz_points)

width_ana = pulse.calc_analytical_pulse_width_over_propagation_distance(
    zz, pulse_std, beta[2], time_units="pico", space_units="kilo", plot=False)

# width_num_ana = pulse._calc_numerically_width_spread_from_analytical_pulse_shape(zz, pulse_std, alpha, beta, ng, plot=False)

zz_fft = np.array([])
fwhm_fft = np.array([])

plt.loglog(zz/1e3, width_ana/1e-12, label="Solução analítica")
if len(fwhm_fft) > 0:
    plt.loglog(zz_fft/1e3, fwhm_fft/1e-12, "ro", label="Solução numérica (FFT)", markersize=4)
# plt.loglog(zz, width_num_ana)
plt.legend()
plt.xlabel("Distância de propagação (z) [km]")
plt.ylabel("Largura do pulso (FWHM) [ps]")
plt.minorticks_on()
plt.grid(which="both")
plt.xlim(zi/1e3, zf/1e3)
plt.ylim(10, 10e3)
plt.savefig("propagated_FWHM.pdf")
plt.show()