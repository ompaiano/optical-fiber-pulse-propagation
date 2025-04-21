"""Stuff for dispersion analysis of optical pulses.
Depends on the 'propagation' module.

To do:
    1. Check if:
        a. Shouldn't we be able to calculate alpha's frequency dependence from beta by Kramers-Kronig relations?
"""
import numpy as np
from scipy.constants import speed_of_light, nu2lambda
from scipy.constants import micro, nano, pico, kilo, mega, giga
import matplotlib.pyplot as plt

# import propagation
from utils import std2fwhm, calc_pdf_std, prefix_units

def gaussian_pulse_electric_field_propagation(t, z, f0, time_domain_std, beta, **kwargs):
    """See Okamoto's 'Fundamentals of Optical Waveguides', page 84.
    """
    alpha     = kwargs["alpha"] if "alpha" in kwargs else 0.0
    amplitude = kwargs["amplitude"] if "amplitude" in kwargs else 1.0
    phase     = kwargs["phase"] if "phase" in kwargs else 0.0
    w0 = 2*np.pi*f0
    tin = np.sqrt(2.0) * time_domain_std
    delta_t = 2.0*beta[2]*z/tin
    tout = np.sqrt(tin**2 + delta_t**2)
    aux1 = ((t-beta[1]*z)/tout)**2
    aux2 = delta_t/tin
    theta = aux2 * aux1 - .5*np.arctan(aux2)
    A = amplitude / np.sqrt(tout/tin)
    f = A * np.exp( -alpha*z - aux1 + 1j*(w0*t - beta[0]*z + theta) ) 
    return f

def calc_analytical_pulse_width_over_propagation_distance(
    zz: float or array_like, 
    initial_pulse_std: float,
    beta2: float, 
    savepath: None or str = None, 
    width_type: str = "FWHM", 
    plot: bool = True, 
    **kwargs):
    tin = np.sqrt(2.0) * initial_pulse_std
    delta_t = np.abs(2*beta2*zz/tin)
    stds_out = np.sqrt( tin**2 +  delta_t**2 ) / np.sqrt(2.0)
    if width_type == "std":
        widths = stds_out
    elif width_type == "FWHM":
        widths = std2fwhm(stds_out)

    if plot:
        time_prefix, time_prefix_str, space_prefix, space_prefix_str = [None]*4
        if "time_units" in kwargs:
            time_prefix = prefix_units(kwargs["time_units"])
            time_prefix_str = prefix_units(kwargs["time_units"], return_type="str")
        else:
            time_prefix = 1
            time_prefix_str = ""
        if "space_units" in kwargs:
            space_prefix = prefix_units(kwargs["space_units"])
            space_prefix_str = prefix_units(kwargs["space_units"], return_type="str")
        else:
            space_prefix = 1
            space_prefix_str = ""
        plt.loglog(zz/space_prefix, widths/time_prefix)
        plt.xlabel("Propagation distance [{}m]".format(space_prefix_str))
        plt.ylabel("Pulse width ({}) [{}s]".format(width_type, time_prefix_str))
        plt.minorticks_on()
        plt.grid(which="both")
        if "xlim" in kwargs:
            plt.xlim(*kwargs["xlim"])
        if "ylim" in kwargs:
            plt.ylim(*kwargs["ylim"])
        if savepath is None:
            savepath = f"analytical_FWHM_beta2{beta2}.pdf"
        plt.savefig(savepath)
        plt.show()
    return widths

def _calc_numerically_width_spread_from_analytical_pulse_shape(
    zz: float or array_like, 
    initial_pulse_std: float,
    alpha: float, 
    beta: np.array,
    ng: float, 
    f0: float = 193e12,
    width_type: str = "FWHM", 
    tt: None or np.array = None, 
    plot: bool = True, 
    savepath: str or None = None):
    """Numerically calculate the width spread from the ANALYTICAL Gaussian 
    pulse shape for different distances.

    This is mainly to check if I implemented the analytical solution correctly.
    """
    vg = speed_of_light/ng
    if tt is None:
        ti = -100*initial_pulse_std
        tf = 100*initial_pulse_std
        num_tt_points = int(1e5)
        tt = np.linspace(ti, tf, num_tt_points)
    widths = []
    if not hasattr(zz, "__len__"):
        zz = [zz]
    for zi in zz:
        time_displacement = zi/vg
        closed_sol = gaussian_pulse_electric_field_propagation(
            tt+time_displacement, zi, f0, initial_pulse_std, beta)
        w = calc_pdf_std(tt, np.abs(closed_sol))
        if width_type == "FWHM":
            w = 2*np.sqrt(2*np.log(2))*w
        widths.append(w)
    if plot:
        plt.loglog(zz, widths)
        # plt.xlabel()
        # plt.ylabel()
        plt.minorticks_on()
        plt.grid(which="both")
        if savepath is None:
            savepath = f"numerical_width_spread_from_analytical_pulse_evolution_beta2{beta[2]}.pdf"
        plt.savefig(savepath)
        plt.show()
    return widths


class pulse:
    def __init__(self, f0, beta, alpha, time=None, pulse_shape=None):
        """Class pulse.
        
        Parameters
        ----------
        f0: float
            Central linear frequency, in Hertz [Hz].

        pulse_std_on_time_domain: float
            Pulse standard deviation on time domain, in seconds [s].

        beta: array_like[float]
            
        alpha: array_like[float] or float
            Waveguide attenuation [dB/m].
            For optical fibers, alpha is approximately 1.77e-6 [m^-1].
            Shouldn't we be able to calculate alpha's frequency dependence from beta by Kramers-Kronig relations?

        Returns
        -------
        """
        self.f0 = f0
        self.w0 = 2.0*np.pi*f0
        self.beta = beta
        self.alpha = alpha
    
    def calc_pulse_spectrum():
        pass

    def propagate_pulse(z):
        pass

    def plot_pulse_shape():
        pass

    def plot_pulse_spectrum():
        pass


class gaussian_pulse(pulse):
    def __init__(self, f0: float, beta: list[float], time_domain_std: float,  alpha: float = 0.0):
        """Class gaussian pulse.
        """
        super().__init__(f0, alpha)
        self.time_domain_std = time_domain_std