 # -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:18:36 2020

@author: giuseppe
"""

import numpy as np
import os, fnmatch

def closest(array, n):
    """
    Find index of the element in the array that is the closest to the value n.
    """
    idx = (np.abs(array - n)).argmin() 
    return idx

def kev2ang(energy):
    """
    Convert energy [keV] to wavelength [Å]
    """
    wlen = 12.398/energy
    return wlen
    
def attenuation(beta, energy):
    """
    Calculate attenuation, mu, given the material constant beta and the energy.
    """
    wlen = kev2ang(energy)*1e-10
    k = 2*np.pi/wlen
    mu = 2*k*beta
    return mu

def trapzint(x, f, x0, x1):
    """
    Integrate a function f with the trapezoid method.
    """
    trapz_sum = 0
    idx = closest(x, x0)
    fdx = closest(x, x1)
    dx = x[1] - x[0]
    for i in range(idx, fdx - 1):
        trapz_sum += (f[i] + f[i + 1])*(dx/2)
    return trapz_sum

def normalize(y, n):
    """
    Normalize a function y to a value n.
    """
    norm = (y - np.min(y))/(np.max(y) - np.min(y))
    return norm*n

def fwhm(x, y):
    """
    Measure the FWHM of a peak.
    """
    peak = np.max(y)
    i_peak = np.argmax(y)
    hm = peak/2                     # half maximum
    i0 = closest(y[:i_peak], hm)
    i1 = closest(y[i_peak:], hm) + i_peak
    fwhm = x[i1] - x[i0]
    return fwhm

def voigt_fwhm(gamma, sigma):
    """
    Calculate the FWHM of a Voigt profile given gamma and sigma.
    """
    fL = 2*gamma
    fG = 2*sigma*np.sqrt(2*np.log(2))
    fV = np.power(fG**5 + 2.69269*fG**4*fL + 2.42843*fG**3*fL**2 + 4.47163*fG**2*fL**3 + 0.07842*fG*fL**4 + fL**5,1/5)
    return fV
    
class finder:
    """
    Find a file with a certain filename.
    """
    def __init__(self, pattern, path):
        self.result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    self.result.append(os.path.join(root, name))
                    
    def find(self):
        return self.result[0]