#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:17:22 2021

@author: giuseppe
"""


import numpy as np
from PIL import Image
from pyLjus.utils import closest, finder

def angle_maps(gamma, delta, ci, cj, w, h, SDD, pxl_size, ph):
    """
    Angle calculations for a (2+3) diffractometer and an area detector
    with following parameters:
        
    gamma = nominal detector angle gamma [degrees]
    delta = nominal detector angle delta [degrees]
    ci, cj = direct beam (x,z) coordinates [pixels]
    w, h = detector width and height [pixels]
    SDD = sample-to-detector distance [m]
    pxl_size = pixel size [m]
    ph = amount of horizontal polarization [0<ph<1]
    """
    gamma_map = np.empty((h,w))                 # initialize detector gamma map
    delta_map = np.empty((h,w))                 # initialize detector delta map
    d = np.empty((h,w))                      # initialize detector distance map
    corr_i = np.empty((h,w))          # initialize flat detector correction map
    g_offset = (-1.08435537e-6*gamma**2 - 
                0.00084077357*gamma - 
                0.0128920777)                      # gamma offset (calibration)
    gamma += g_offset                                  # correct gamma position
    d_offset = (1.7280529238e-6*delta**2 - 
                0.000700361461*delta - 
                0.00367551936)                     # delta offset (calibration)
    delta += d_offset                                  # correct delta position
    nom_gamma = np.deg2rad(gamma)         # convert nominal det angles to [rad]
    nom_delta = np.deg2rad(delta)         # convert nominal det angles to [rad]
    GAM = np.array([[np.cos(nom_gamma),np.sin(nom_gamma),0],                
                     [-np.sin(nom_gamma), np.cos(nom_gamma),0], 
                     [0,0,1]])                       # \Gamma rotational matrix
    DEL = np.array([[1,0,0],                         # \Delta rotational matrix
                    [0,np.cos(nom_delta),-np.sin(nom_delta)], 
                    [0,np.sin(nom_delta),np.cos(nom_delta)]])
    rot_mat = np.matmul(GAM,DEL)                 # multiply rotational matrices
    for j in range(h):
        dz = (cj - j)*pxl_size          # delta z (z-distance from det. center)
        for i in range(w):
            dx = (ci - i)*pxl_size      # delta x (x-distance from det. center)
            di = np.sqrt(dx**2 + SDD**2 + dz**2)     # sample-to-pixel distance
            dr = np.sqrt(dx**2 + dz**2)              # center-to-pixel distance
            p = np.array([dx, SDD, dz])           # central pixel position at
                                           # zero angles in the lab coordinates
            (xp, yp, zp) = np.matmul(rot_mat, p)    # central pixel position at
                                                    # nominal detector angle
            gamma_map[j][i] = np.arctan(xp/yp)      # map of gamma pixel values
            delta_map[j][i] = np.arcsin(zp/di)      # map of delta pixel values
            d[j][i] = di                            # map of SDD distances
            corr_i[j][i] = 1/(np.cos(np.arctan(dr/SDD)))     # flat det. corr.
    corr_d = np.power(d,2)/np.power(SDD,2)                   # flat det. corr.
    chi = np.arctan(np.tan(delta_map)/np.tan(gamma_map))     # map of chi
    Phor = (1 - 
            np.power(np.sin(gamma_map),2))          # horizontal component of 
                                                    # polarization correction
    Pver = (1 - 
            np.power(np.sin(delta_map)*np.cos(gamma_map),2)) # vertical comp.
                                                # of polarization correction
    P = ph*Phor + (1-ph)*Pver                        # polarization correction
    tth = np.arccos(np.cos(delta_map)*np.cos(gamma_map))     # 2th map
    L = 1/(np.sin(tth/2)*np.sin(tth))                      # Lorentz correction
    flat = corr_i * corr_d                              # flat det. correction
    PL = P * L * flat                                   # multiply corrrections
    return tth, chi, PL

def angle_maps_slits(gamma, delta, ci, cj, w, h, SDD, pxl_size, ph, Rs):
    """
    Angle calculations for a (2+3) diffractometer, an area detector and slits.
    Same calcualtions as above but here
    SDD = slit-to-detector distance
    Rs = sample-to-slit distance
    """
    gamma_map = np.empty((h,w))                 # initialize detector gamma map
    delta_map = np.empty((h,w))                 # initialize detector delta map
    d = np.empty((h,w))                      # initialize detector distance map
    corr_i = np.empty((h,w))          # initialize flat detector correction map
    g_offset = (-1.08435537e-6*gamma**2 - 
                0.00084077357*gamma - 
                0.0128920777)                      # gamma offset (calibration)
    gamma += g_offset                                  # correct gamma position
    d_offset = (1.7280529238e-6*delta**2 - 
                0.000700361461*delta - 
                0.00367551936)                     # delta offset (calibration)
    delta += d_offset                                  # correct delta position
    nom_gamma = np.deg2rad(gamma)         # convert nominal det angles to [rad]
    nom_delta = np.deg2rad(delta)         # convert nominal det angles to [rad]
    GAM = np.array([[np.cos(nom_gamma),np.sin(nom_gamma),0],                
                     [-np.sin(nom_gamma), np.cos(nom_gamma),0], 
                     [0,0,1]])                       # \Gamma rotational matrix
    DEL = np.array([[1,0,0],                         # \Delta rotational matrix
                    [0,np.cos(nom_delta),-np.sin(nom_delta)], 
                    [0,np.sin(nom_delta),np.cos(nom_delta)]])
    rot_mat = np.matmul(GAM,DEL)                 # multiply rotational matrices
    for j in range(h):
        dz = (cj - j)*pxl_size          # delta z (z-distance from det. center)
        for i in range(w):
            dx = (ci - i)*pxl_size      # delta x (x-distance from det. center)
            di = np.sqrt(dx**2 + (SDD + Rs)**2 + 
                         dz**2)                     # sample-to-pixel distance
            dr = np.sqrt(dx**2 + dz**2)              # center-to-pixel distance
            s = np.array([0, Rs, 0])                # sample-to-slit vector
            (xs, ys, zs) = np.matmul(rot_mat, s)    # rotate s vector
            p = np.array([dx, (SDD + Rs), dz])      # central pixel position at
                                           # zero angles in the lab coordinates
            (xp, yp, zp) = np.matmul(rot_mat, p)    # central pixel position at
                                                    # nominal detector angle
            dps = np.sqrt((xp - xs)**2 + (yp - ys)**2 + 
                          (zp - zs)**2)                # pixel-to-slit distance
            gamma_map[j][i] = np.arctan((xp - xs)/(yp - ys))        # gamma map
            delta_map[j][i] = np.arcsin((zp - zs)/dps)              # delta map
            d[j][i] = di                            # map of SDD distances
            corr_i[j][i] = 1/(np.cos(np.arctan(dr/SDD)))     # flat det. corr.
    corr_d = np.power(d,2)/np.power(SDD,2)                   # flat det. corr.
    chi = np.arctan(np.tan(delta_map)/np.tan(gamma_map))     # map of chi
    Phor = (1 - 
            np.power(np.sin(gamma_map),2))          # horizontal component of 
                                                    # polarization correction
    Pver = (1 - 
            np.power(np.sin(delta_map)*np.cos(gamma_map),2)) # vertical comp.
                                                # of polarization correction
    P = ph*Phor + (1-ph)*Pver                        # polarization correction
    tth = np.arccos(np.cos(delta_map)*np.cos(gamma_map))     # 2th map
    L = 1/(np.sin(tth/2)*np.sin(tth))                      # Lorentz correction
    flat = corr_i * corr_d                              # flat det. correction
    PL = P * L * flat                                   # multiply corrrections
    return tth, chi, PL

def scan2plot(datafolder, start, end, first, last,
                 theta_range, theta_bins, chi_range, chi_bins,
                 gamma, delta,
                 ci, cj, w, h, SDD, pxl_size, ph, d5i=None,
                 fraction=1):
    """
    Combine images from a detector scan in a 2d plot of intensity vs (2th, chi).
    datafolder = folder containing the images [string]
    start, end = indeces of the first and last image of the scan [integers]
    first, last = indeces of the first and last image that you want to 
        process [start<first, last<end]
    theta_range = tuple with first and last 2th values [degrees]
    theta_bins = number of bins in the resulting 2D plot
    chi_range = tuple with first and last chi values [degrees]
    chi_bins = number of chi bins in the resulting 2D plot
    gamma, delta = lists of nominal detector angles for each image of the scan
    ci, cj = direct beam (x,z) coordinates [pixels]
    w, h = detector width and height [pixels]
    SDD = sample-to-detector distance [m]
    pxl_size = pixel size [m]
    ph = amount of horizontal polarization [0<ph<1]
    d5i = list containing the normalization factors
            (i.e., ring current or other corrections)
            No corrections by default.
    fraction = amount of detector used in the processing [0<fraction<1]
    """
    chi_bins = int(chi_bins)                # make sure the input is an integer
    theta_bins = int(theta_bins)            # make sure the input is an integer
    chi_ax = np.linspace(chi_range[0], 
                         chi_range[1], chi_bins)                # init chi axis
    tth_ax = np.linspace(theta_range[0], 
                         theta_range[1], theta_bins)            # init 2th axis
    int_bin = np.zeros((chi_bins, theta_bins))            # init intensity plot
    tth_weight = np.zeros(theta_bins)               # init weight normalization
                    # (i.e., the number of times a certain bin has been filled)
    for i in range(first, last + 1):
        print("delta = " + str(delta[i - start]) + ", gamma = " + 
              str(gamma[i - start]) + 
              ", status: " + str(i  - start) + "/" + 
              str(last - first))                 # print info on current status
        fname = finder("*" + str(i) + 
                       ".tif", datafolder).find()   # find image with index i
        with Image.open(fname) as img:
            tth_map, chi_map, PL = angle_maps(gamma[i - start], 
                                              delta[i - start], 
                                              ci, cj, w, h, 
                    SDD, pxl_size, ph)                     # angle calculations
            det_img = np.array(img)              # convert image to numpy array
            if d5i.any() != None:
                det_img = det_img/(d5i[i - start])  # normalize data to monitor
            det_img /= PL                           # correct by Lorentz-pol.
            # data binning:
            for j in range(int(h/2*(1-fraction)), int(h/2*(1+fraction))):
                for k in range(int(w/2*(1-fraction)), int(w/2*(1+fraction))):
                    # find bin on the 2th axis
                    idx = closest(tth_ax, np.rad2deg(tth_map[j][k]))
                    # find bin on the chi axis
                    jdx = closest(chi_ax, np.rad2deg(chi_map[j][k]))
                    # fill bin
                    int_bin[jdx][idx] += det_img[j][k]
                    # every time a bin is filled add 1 to the weight function
                    tth_weight[idx] += 1
    print("Done!")
    return tth_ax, chi_ax, int_bin, tth_weight

def plot2pattern(tth_ax, int_bin, tth_weight):
    """
    Integrate the 2D plot to produce a powder diffraction pattern.
    """
    tth_integral = np.zeros(len(tth_ax))          # initialize an empty pattern
    for row in int_bin:                          # for every row of the 2D plot
        tth_integral += row             # sum every row into the integral array
    tth_integral /= tth_weight                     # correct by weight function
    return tth_integral