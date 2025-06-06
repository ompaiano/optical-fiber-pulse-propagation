import numpy as np
import scipy as sp
import scipy.constants as cts
import matplotlib.pyplot as plt

from utils import std2fwhm, fwhm2std, calc_pdf_mean, calc_pdf_std
import pulse

# Pulse temporal shape:
def gaussian_pulse(time, std, mean=0.0):
    x = time - mean
    pulse = np.exp(-.5*(x/std)**2)
    return pulse

def train_of_pulses(std, delay, num_pulses, num_points, bits=None):
    if bits is None:
        bits = np.ones(num_pulses)
    total_time_interval = delay * num_pulses
    ti = -total_time_interval/2.0
    tf = +total_time_interval/2.0
    time = np.linspace(ti, tf, num_points)
    train = np.zeros(num_points)
    pulses = []
    for n, b in enumerate(bits):
        if b >= .5:
            pulse_position = ti + delay/2.0 + n*delay
            pulses.append(gaussian_pulse(time, std, pulse_position))
            train += pulses[-1]
    return train, pulses, time

# Pulse spectrum for real signal:
def calc_spectrum_real_signal(signal, sample_rate, plot=True, xlim=(1.90e14, 1.95e14)):
    spectrum = sp.fft.rfft(sp.fft.fftshift(electric_field))
    freq = sp.fft.rfftfreq(npts, d=sample_rate)
    if plot:
        plt.semilogx(freq, spectrum.real)
        if xlim is not None:
            plt.xlim(*xlim)
        plt.show()
    return spectrum, freq

# Pulse spectrum for complex signal:
def calc_spectrum(signal, sample_rate, plot="linear", step=5, xlim=(1.90e14, 1.95e14), ylim=None):
    spectrum = sp.fft.fft(sp.fft.fftshift(electric_field))
    freq = sp.fft.fftfreq(npts, d=sample_rate)
    spectrum = np.fft.ifftshift(spectrum)
    freq = np.fft.ifftshift(freq)
    if plot is not None:
        fig, ax_real = plt.subplots(figsize=(8, 4))
        ax_imag = ax_real.twinx()
        if plot == "linear":
            ax_real.plot(freq[::step], spectrum.real[::step])
            ax_imag.plot(freq[::step], spectrum.imag[::step], '--', color='orange')
        elif plot == "semilogx":
            ax_real.semilogx(freq[::step], spectrum.real[::step])
            ax_imag.semilogx(freq[::step], spectrum.imag[::step], '--', color='orange')
        elif plot == "semilogy":
            ax_real.semilogy(freq[::step], spectrum.real[::step])
            ax_imag.semilogy(freq[::step], spectrum.imag[::step], '--', color='orange')
        elif plot == "loglog":
            ax_real.loglog(freq[::step], spectrum.real[::step])
            ax_imag.loglog(freq[::step], spectrum.imag[::step], '--', color='orange')
        if xlim is not None:
            ax_real.set_xlim(*xlim)
        if ylim is not None:
            pass
        else:
            ax_imag.set_ylim(ax_real.get_ylim())
        ax_real.set_xlabel("Frequency [Hz]")
        ax_real.set_ylabel("Real amplitude")
        ax_imag.set_ylabel("Imaginary amplitude")
        plt.show()
        plt.close()
    return spectrum, freq

# Functions for pulse propagation:
def taylor_approx(span, coeffs, center=0.0):
    """Calculate the Taylor expansion over a span given the coefficients.
    The coefficients are expected to correspond to angular frequency.

    To do: dilute the factorial to avoid overflow for degree higher than 70.

    Parameters
    ----------
    span: array
        Taylor expansion domain.
    coeffs: array
        n-th order derivative at point "center".

    center: float
        Center of the Taylor expansion.

    Returns
    -------
    numpy array.
        Taylor approximation over the requested span.
    """
    approx = np.zeros(len(span))
    degree = np.arange(len(coeffs))
    for ni, ci in zip(reversed(degree), reversed(coeffs)):
        approx += ci*(span-center)**ni / sp.special.factorial(ni)
    return approx

def approx_wavenumber(freq, kn, center_freq, window=None, ang_freq=True, plot=False):
    """Calculate the Taylor expansion of $k(\\omega)$.
    Frequency may be linear or angular, provided the coefficients are alike.
    Alternatively, the parameter ang_freq converts linear frequency to angular
    frequency if set to False.

    Parameters
    ----------
    freq: array
        Frequency span.
    kn: array
        List of coefficients for the Taylor expansion. kn[0] must be the
    center_freq: float
        Center frequency: where the coefficients kn must have been derived at.
    ang_freq: bool
        Converts linear frequency to angular frequency if set to False.

    Returns
    -------
    numpy array.
        Taylor approximation for the wavenumber as a function of frequency.
    """
    if not ang_freq:
        freq *= 2.0*np.pi
        center_freq *= 2.0*np.pi
        if window is not None:
            window *= 2.0*np.pi

    if window is not None:
        freq_inf = center_freq - window/2
        freq_sup = center_freq + window/2
        idx_inf = np.searchsorted(freq, freq_inf)
        idx_sup = np.searchsorted(freq, freq_sup)
        window_freq = freq[idx_inf:idx_sup]
        approx = np.zeros(len(freq))
        approx[idx_inf:idx_sup] = taylor_approx(window_freq, kn, center=center_freq)
    else:
        approx = taylor_approx(freq, kn, center=center_freq)
    if plot:
        plt.plot(freq, approx)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Wavenumber [m^-1]")
        if window is not None:
            pass
            # plt.xlim(freq_inf, freq_sup)
        plt.show()

    return approx

def propagation_operator(z, k):
    """

    Parameters
    ----------
    z: float
        Propagation distance.
    k: array_like
        Wavenumber angular frequency response.

    Returns
    -------

    """
    return np.exp(-1j*np.array(k)*z)

def time_shift_operator(frequency, t0, ang_freq=False):
    """Time shifting operator (acts on frequency domain).
    Useful for retrieving the pulse centered at a specific time.
    Usually we want t0 = z/vg.
    """
    if ang_freq:
        ang_const = 1
    else:
        ang_const = 2.0*np.pi
    return np.exp(1j*ang_const*frequency*t0)

def propagate_pulse(freq, spectrum, operator, time_shift=None):
    propagation_on_freq_domain = spectrum * operator
    if time_shift is not None:
        propagation_on_freq_domain *= time_shift_operator(freq, time_shift)
    propagation_on_time_domain = sp.fft.ifft(
        sp.fft.fftshift(propagation_on_freq_domain))
    time = sp.fft.fftfreq(len(freq), d=freq[1]-freq[0])
    return sp.fft.ifftshift(propagation_on_time_domain), sp.fft.ifftshift(time)


# Parameters:
wavelength = 1.55e-6 # meters;
freq0 = cts.lambda2nu(wavelength) # Hertz;
w0 = 2*np.pi*freq0
pulse_FWHM = 20e-12  # seconds;
pulse_std = fwhm2std(pulse_FWHM)
k = np.array([6.0e6, 5.0e3, -2.0e-2]) * np.array([1.0, 1e-12, 1e-24]) # rad/m; s/m; s^2/m.


# Calculating the electric field:
npts = int(4e7)
ti = -1000e-12 # seconds; 5 nanometros
tf = +1000e-12 # seconds;
time = np.linspace(ti, tf, npts) #, dtype=np.float32)
sample_rate = time[1] - time[0] # seconds;
pulse_shape = np.array(gaussian_pulse(time, pulse_std))
print(type(pulse_shape[0]))
electric_field = pulse_shape * np.exp(+1j*w0*time) 
print(type(electric_field[0]))
plt.plot(time[::4], electric_field[::4].real)
plt.plot(time[::4], pulse_shape[::4])
plt.show()
plt.close()

# Pulse spectrum:
spectrum, freq = calc_spectrum(electric_field, sample_rate, plot="linear", xlim=None, step=4)

# Pulse propagation of a distance of z:
z = 20e3 # meters.
vg = 1/k[1]
time_delay = z/vg
print(f"Pulse time delay after propagation of z = {z} meters: \t\t {time_delay} seconds.")


wavenumber_freq = approx_wavenumber(2*np.pi*freq, k, 2*np.pi*freq0, window=100e12, plot=False)
propagator = propagation_operator(z=z, k=wavenumber_freq)
pulse, time = propagate_pulse(freq, spectrum, propagator, time_delay)

# Plotting the pulse's temporal shape after a propagation of z:
plt.plot(time, np.abs(pulse)**1)
plt.show()
plt.close()

# Pulse parameters:
dt = time[1] - time[0]
normalization_constant = np.abs(pulse).sum() * dt
pulse_pdf = np.abs(pulse) / normalization_constant
z_propagated_pulse_mean = (pulse_pdf * time * dt).sum()
z_propagated_pulse_var = ( pulse_pdf * dt * (time-z_propagated_pulse_mean)**2 ).sum()
z_propagated_pulse_std = np.sqrt(z_propagated_pulse_var)
z_propagated_pulse_FWHM = 2.0*np.sqrt(2.0*np.log(2.0)) * z_propagated_pulse_std

print(f"Pulse mean after propagation of z={z} meters: \t\t", z_propagated_pulse_mean, " seconds.")
print(f"Pulse std deviation after propagation of z={z} meters: \t", z_propagated_pulse_std, " seconds.")
print(f"Pulse FWHM after propagation of z={z} meters: \t\t", z_propagated_pulse_FWHM, " seconds.")