import numpy as np

from scipy.constants import pico, nano, micro, milli, kilo, mega, giga, tera


def std2fwhm(std):
    return 2.0*np.sqrt(2.0*np.log(2.0))*std

def fwhm2std(fwhm):
    return fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))

def calc_pdf_mean(x, f, normalized=False, equidistant_points=True):
    if not equidistant_points:
        raise NotImplementedError()
    x, pdf = np.array(x), np.array(f)
    dx = x[1] - x[0]
    if not normalized:
        norm_c = pdf.sum() * dx
        pdf /= norm_c
    return (pdf * x * dx).sum()

def calc_pdf_std(x, f, normalized=False, equidistant_points=True, **kwargs):
    if not equidistant_points:
        raise NotImplementedError()
    x, pdf = np.array(x), np.array(f)
    dx = x[1] - x[0]
    if not normalized:
        norm_c = pdf.sum() * dx
        pdf /= norm_c
    if "mean" in kwargs:
        mean = kwargs["mean"]
    else:
        mean = calc_pdf_mean(x, pdf, normalized=True)
    variance = ( pdf * dx * (x-mean)**2 ).sum()
    return np.sqrt(variance)

def prefix_units(prefix: str, return_type="float"):
    if prefix == "pico":
        return pico if return_type == "float" else "p"
    elif prefix == "nano":
        return nano if return_type == "float" else "n"
    elif prefix == "micro":
        return micro if return_type == "float" else "$\\mu$"
    elif prefix == "milli":
        return milli if return_type == "float" else "m"
    elif prefix == "kilo":
        return kilo if return_type == "float" else "k"
    elif prefix == "mega":
        return mega if return_type == "float" else "M"
    elif prefix == "giga":
        return giga if return_type == "float" else "G"
    elif prefix == "tera":
        return tera if return_type == "float" else "T"
    else:
        return 1.0 if return_type == "float" else ""