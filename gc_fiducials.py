#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gc_fiducials.py
#  
#  Copyright 2014 Greg Green <greg@greg-UX31A>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np

from scipy.interpolate import interp1d, UnivariateSpline
import scipy.interpolate
from scipy.ndimage.filters import gaussian_filter1d
import scipy.special

import decimalpy as dp

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import os, glob


#
# De-reddening
#

reddening_vector_dict = {'u_SDSS':4.239,
                         'g_SDSS':3.303,
                         'r_SDSS':2.285,
                         'i_SDSS':1.698,
                         'z_SDSS':1.263,
                         'g_P1':3.172,
                         'r_P1':2.271,
                         'i_P1':1.682,
                         'z_P1':1.322,
                         'y_P1':1.087,
                         'g_DES':3.237,
                         'r_DES':2.176,
                         'i_DES':1.595,
                         'z_DES':1.217,
                         'Y_DES':1.058,
                         'J':0.786,
                         'H':0.508,
                         'K_s':0.320}


def get_reddening_vector(bands):
    R = [reddening_vector_dict[b] for b in bands]
    return np.array(R)


def dereddened_mags(mags, EBV, bands):
    R = get_reddening_vector(bands)
    if type(EBV) == float:
        R.shape = (1, R.size)
        R = np.repeat(R, len(mags), axis=0)
        return mags - EBV * R
    elif type(EBV) == np.ndarray:
        return mags - np.einsum('i,j->ij', EBV, R)
    else:
        raise TypeError('EBV has unexpected type: %s' % type(EBV))


#
# Fiducials
#

# [Fe/H], DM, E(B-V)
prop_dict = {'NGC188':(-0.03, 11.15, 0.08), #
             'NGC288':(-1.32, 14.84, 0.03),
             'NGC1904':(-1.60, 15.59, 0.01),
             'NGC2682':(0.03, 9.49, 0.04), #
             'NGC4590':(-2.23, 15.21, 0.05),
             'NGC5272':(-1.50, 15.07, 0.01),
             'NGC5897':(-1.90, 15.76, 0.09),
             'NGC5904':(-1.29, 14.46+0.25, 0.03),
             'NGC6205':(-1.53, 14.33, 0.02),
             'NGC6341':(-2.31, 14.65, 0.02),
             'NGC6791':(0.42, 13.51, 0.16), #
             'NGC6838':(-0.78, 13.80-0.20, 0.25),
             'NGC7078':(-2.37, 15.39, 0.10), # Good template
             'NGC7089':(-1.65, 15.50, 0.06),
             'NGC7099':(-2.27, 14.64, 0.03),
             'Pal12':(-0.85, 16.46, 0.02)} #


def read_ascii_table(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    
    lines = [l.lstrip().rstrip() for l in lines]
    data = []
    
    for l in lines:
        if l.startswith('#') or (len(l) == 0):
            continue
        
        data.append([float(x) for x in l.split()])
    
    return np.array(data)


def load_fiducials(selection=None):
    base_path = "/home/greg/projects/stellar-locus/data/Edouard's GCs/"
    
    fnames = []
    
    #selection = [7078, 6341, 7089, 6205, 5272, 5904, 6838, 6791]
    #selection = [7078, 6838, 6791]
    
    if selection == None:
        fnames = glob.glob(base_path + "fid_*.txt")
    else:
        fnames = [glob.glob(base_path + 'fid_NGC%d_*' % n)[0] for n in selection]
    
    names = [fn.split('_')[1] for fn in fnames]
    props = np.array([list(prop_dict[n]) for n in names])
    FeH, DM, EBV = props.T
    
    props = np.empty(len(names), dtype=[('name','S10'),
                                        ('FeH','f8'),
                                        ('DM','f8'),
                                        ('EBV','f8')])
    props['name'][:] = names
    props['FeH'][:] = FeH
    props['DM'][:] = DM
    props['EBV'][:] = EBV
    
    mags_tmp = [read_ascii_table(fn) for fn in fnames]
    bands = ['g_P1', 'r_P1', 'i_P1', 'z_P1', 'y_P1']
    mags = [dereddened_mags(m, float(E), bands) for m, E in zip(mags_tmp, EBV)]
    
    for p in props:
        print p['name'], p['FeH'], p['EBV']
    
    return props, mags


def plot_fiducials(props, mags):
    b1 = 0
    bands = 'grizy'
    
    mplib.rc('text', usetex=True)
    
    #
    # Color-magnitude diagram
    #
    
    fig = plt.figure(figsize=(10,6), dpi=100)
    
    axgrid = Grid(fig, 211,
                  nrows_ncols=(1,4),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    F_0, F_1 = np.min(props['FeH']), np.max(props['FeH'])
    DF = F_1 - F_0
    
    jet = plt.get_cmap('jet')
    c_norm = mplib.colors.Normalize(vmin=F_0, vmax=F_1)
    sm = mplib.cm.ScalarMappable(norm=c_norm, cmap=jet)
    sm._A = []
    
    for b2,b_name in enumerate(bands):
        if b2 == 0:
            continue
        
        ax = axgrid[b2-1]
        fig.add_axes(ax)
        
        for p,m in zip(props, mags):
            c = sm.to_rgba(p['FeH'])
            x = m[:,b1]-m[:,b2]
            idx = np.isfinite(x)
            
            ax.plot(x[idx], m[idx,1]-p['DM'], c=c)
            #print p['name'], np.min(m[:,1]-p['DM'])
        
        ax.set_xlabel(r'$g - %s$' % (b_name), fontsize=16)
        ax.set_ylabel(r'$r$', fontsize=16)
        
        if b2 == 4:
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[1], ylim[0])
    
    axgrid = Grid(fig, 212,
                  nrows_ncols=(1,4),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    for b2,b_name in enumerate(bands):
        if b2 == 0:
            continue
        
        ax = axgrid[b2-1]
        fig.add_axes(ax)
        
        for p,m in zip(props, mags):
            c = sm.to_rgba(p['FeH'])
            x = m[:,b2-1]-m[:,b2]
            idx = np.isfinite(x)
            
            ax.plot(x[idx], m[idx,1]-p['DM'], c=c)
            #print p['name'], np.min(m[:,1]-p['DM'])
        
        ax.set_xlabel(r'$%s - %s$' % (bands[b2-1], bands[b2]), fontsize=16)
        ax.set_ylabel(r'$r$', fontsize=16)
        
        if b2 == 4:
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[1], ylim[0])
    
    fig.suptitle(r'$\mathrm{GC \ Fiducials}$', fontsize=18)
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.10, top=0.90)
    cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])
    cbar = fig.colorbar(sm, orientation='vertical', cax=cax)
    cax.set_ylabel(r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', fontsize=14)
    
    fig.savefig('GC-fiducials.png', dpi=100)
    
    #
    # Turnoff vs. Metallicity
    #
    
    fig = plt.figure(figsize=(20,10), dpi=100)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(1,4),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    for b2,b_name in enumerate(bands):
        if b2 == 0:
            continue
        
        ax = axgrid[b2-1]
        fig.add_axes(ax)
        #ax = fig.add_subplot(1,4,b2)
        
        c = [m[:,b1] - m[:,b2] for m in mags]
        c_MSTO = [np.nanmin(x) for x in c]
        
        coeff = np.polyfit(props['FeH'], c_MSTO, deg=2)
        x = np.linspace(F_0-0.25, F_1+0.25, 1000)
        y = coeff[2] + coeff[1]*x + coeff[0]*x**2.
        ax.plot(x, y, 'k--', lw=2, alpha=0.5)
        
        #spl = UnivariateSpline(props['FeH'], c_MSTO, s=1000000.)
        #y = spl(x)
        #ax.plot(x, y, 'k-', lw=2, alpha=0.5)
        
        ax.scatter(props['FeH'], c_MSTO)
        
        for n,x,y in zip(props['name'], props['FeH'], c_MSTO):
            ax.text(x+0.02, y-0.0025, r'$\mathrm{%s}$' % n, ha='left', va='top', fontsize=8)
    
    for b2,b_name in enumerate(bands):
        if b2 == 0:
            continue
        
        ax = axgrid[b2-1]
        
        c = [m[:,b1] - m[:,b2] for m in mags]
        c_MSTO = [np.nanmin(x) for x in c]
        coeff = np.polyfit(props['FeH'], c_MSTO, deg=2)
        #print coeff
        
        txt = r'$x_{\mathrm{MSTO}} = '
        for k,a in enumerate(coeff[:-2]):
            txt += r'%.3f \, x^{%d} + ' % (a, len(coeff)-k-1)
        txt += r'%.3f \, x ' % (coeff[-2])
        if len(coeff) > 2:
            txt += r'+ '
        txt += r'%.3f$' % (coeff[-1])
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        w = xlim[1] - xlim[0]
        h = ylim[1] - ylim[0]
        x = xlim[0] + 0.05*w
        y = ylim[1] - 0.05*h
        
        ax.text(x, y, txt, ha='left', va='top', fontsize=12)
        
        ax.set_ylabel(r'$\mathrm{color}$', fontsize=18)
        ax.set_xlabel(r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', fontsize=18)
        ax.set_title(r'$g - %s$' % (b_name), fontsize=20)
    
    fig.suptitle(r'$\mathrm{MSTO \ Position}$', fontsize=24)
    fig.subplots_adjust(left=0.07, right=0.95)
    
    fig.savefig('GC-MSTO.png', dpi=100)
    
    #
    # Turnoff vs. Metallicity (color-color)
    #
    
    fig = plt.figure(figsize=(10,10), dpi=100)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(3,3),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    for row in xrange(3):
        i_1 = row+1
        c_1 = [m[:,i_1] - m[:,i_1+1] for m in mags]
        
        for col in xrange(row+1):
            i_2 = col
            c_2 = [m[:,i_2] - m[:,i_2+1] for m in mags]
            
            ax = axgrid[3*row + col]
            fig.add_axes(ax)
            
            for cc_1, cc_2, p in zip(c_1, c_2, props):
                c = sm.to_rgba(p['FeH'])
                idx = np.isfinite(cc_1) & np.isfinite(cc_2)
                ax.plot(cc_2[idx], cc_1[idx], c=c)
                #print p['name'], np.min(cc_1[idx])
    
    fig.savefig('GC-color-color.png', dpi=100)
    
    plt.show()


def interpolate_M(mags, n=5):
    n_FeH, n_M, n_bands = mags.shape
    mags_interp = np.empty((n_FeH, n_M+(n_M-1)*n, n_bands), dtype='f8')
    
    mags_interp[:,::(n+1),:] = mags[:,:,:]
    
    for k in xrange(n):
        a = (k + 1.) / (n + 1.)
        mags_interp[:,k+1:-1:(n+1),:] = (1. - a) * mags[:,:-1,:] + a * mags[:,1:,:]
    
    return mags_interp


def interpolate_FeH(mags, n=5):
    n_FeH, n_M, n_bands = mags.shape
    mags_interp = np.empty((n_FeH+(n_FeH-1)*n, n_M, n_bands), dtype='f8')
    
    mags_interp[::(n+1),:,:] = mags[:,:,:]
    
    for k in xrange(n):
        a = (k + 1.) / (n + 1.)
        mags_interp[k+1:-1:(n+1),:,:] = (1. - a) * mags[:-1,:,:] + a * mags[1:,:,:]
    
    return mags_interp


def interpolate_FeH_extensive(mags, n=5):
    n_FeH, n_M, n_bands = mags.shape
    
    mags_interp = np.empty((n_FeH+(n_FeH+1)*n, n_M, n_bands), dtype='f8')
    
    mags_interp[n::(n+1),:,:] = mags[:,:,:]
    
    
    
    for k in xrange(n):
        a = (k + 1.) / (n + 1.)
        
        mags_interp[n+1+k:-(n+1):(n+1),:,:] = (1. - a) * mags[:-1,:,:] + a * mags[1:,:,:]
        
        mags_interp[k,:,:] = float(n-k) / (n + 1.) * (mags[0,:,:] - mags[1,:,:]) + mags[0,:,:]
        mags_interp[-(k+1),:,:] = float(n-k) / (n + 1.) * (mags[-1,:,:] - mags[-2,:,:]) + mags[-1,:,:]
    
    return mags_interp


def interp_linear(x, n=5):
    n_x = x.shape[0]
    x_interp = np.empty(n_x+(n_x-1)*n, dtype='f8')
    
    x_interp[::(n+1)] = x[:]
    
    for k in xrange(n):
        a = (k + 1.) / (n + 1.)
        x_interp[k+1:-1:(n+1)] = (1. - a) * x[:-1] + a * x[1:]
    
    return x_interp


def interp_linear_extensive(x, n=5, axis=0):
    '''
    Resample an array using linear interoplation along one axis.
    Extends the interpolation off the edge of the original array.
    '''
    
    x_p = np.swapaxes(x, axis, 0)
    
    # Create empty interpolated array
    n_x = x_p.shape[0]
    new_shape = list(x_p.shape)
    new_shape[0] = n_x+(n_x+1)*n
    x_interp = np.empty(new_shape, dtype='f8')
    
    x_interp[n::(n+1),...] = x_p[:,...]
    
    for k in xrange(n):
        a = (k + 1.) / (n + 1.)
        x_interp[n+1+k:-(n+1):(n+1),...] = (1. - a) * x_p[:-1,...] + a * x_p[1:,...]
        
        x_interp[k,...] = float(n-k) / (n + 1.) * (x_p[0,...] - x_p[1,...]) + x_p[0,...]
        x_interp[-(k+1),...] = float(n-k) / (n + 1.) * (x_p[-1,...] - x_p[-2,...]) + x_p[-1,...]
    
    x_interp = np.swapaxes(x_interp, axis, 0)
    
    return x_interp


def B_spline_smooth(x, n=1000, s=0.):
    u = np.linspace(0., 1., x.shape[1])
    tck,u = scipy.interpolate.splprep(x, u=u, s=s)
    
    u = np.linspace(0., 1., n*x.shape[1])
    return scipy.interpolate.splev(u, tck)


def smooth_fiducial(mags, s=0.1, n=10):
    idx = np.all(np.isfinite(mags), axis=1)
    
    print mags[idx,:].shape
    
    return np.array(B_spline_smooth(mags[idx,:].T, n=n, s=float(s*mags.shape[0]))).T


def fit_GCs(plot=False):
    #
    # Load in GC fiducials
    #
    props, mags_tmp = load_fiducials(selection=[7078, 6838, 6791])
    
    mags_tmp = [smooth_fiducial(m, s=0.001) for m in mags_tmp]
    
    n_max = max([m.shape[0] for m in mags_tmp])
    mags = np.empty((3, n_max, 5), dtype='f8')
    
    for i,m in enumerate(mags_tmp):
        mags[i,:m.shape[0],:] = m[:,:] - props['DM'][i]
        mags[i,m.shape[0]:,:] = np.nan
    
    #mags = interp_linear_extensive(mags, n=20, axis=1)
    
    colors = np.empty((mags.shape[0], mags.shape[1], mags.shape[2]-1), dtype='f8')
    
    for k in xrange(colors.shape[2]):
        colors[:,:,k] = mags[:,:,0] - mags[:,:,k+1]
    
    #tmp = np.empty(colors.shape, dtype='f8')
    #for i in xrange(mags.shape[0]):
    #    idx = np.isfinite(colors[i,:,k])
    #    colors[i,:,k] = B_spline_smooth(colors
    
    #mags[1,:,0] = gaussian_filter1d(mags[1,:,0], sigma=5., mode='nearest')
    
    #
    # Determine color offsets as a function of M and [Fe/H]
    #
    
    b1 = 0
    
    c_offset = np.zeros(colors.shape, dtype='f8')
    
    # Loop over M
    for j, (M, c_0) in enumerate(zip(mags[1, :, :], colors[1, :, :])):
        c_0 = np.tile(c_0, (mags.shape[1], 1))
        
        # Loop over [Fe/H]
        for i in [0,2]:
            if M[b1] < np.nanmin(mags[i,:,b1]):
                c_offset[i,j,:] = np.nan
                continue
            
            # Weight each point by distance in M
            dM = mags[i, :, :] - M
            
            '''
            dM2 = np.sum(dM**2, axis=1)
            sigma_dM = 0.05
            w = np.exp(-0.5 * dM2/sigma_dM**2.)
            w *= (np.abs(dM2 < (0.5)**2.)).astype('f8')
            '''
            
            idx = np.sum(dM[:,b1] < 0.)
            w = np.zeros(dM.shape[0], dtype='f8')
            if idx+1 < w.size:
                w[idx+1] = 0.5
            if idx < w.size:
                w[idx] = 1.
            if idx-1 >= 0:
                w[idx-1] = 1.
            if idx-2 >= 0:
                w[idx-2] = 0.5
            w *= (np.abs(dM[:,b1]) < 0.5).astype('f8')
            
            w = np.reshape(np.repeat(w, 4), (w.size,4))
            
            # Determine color offset
            dc = colors[i, :, :] - c_0[:, :]
            
            idx = np.isfinite(dc) & np.isfinite(w)
            w[~idx] = 0.
            dc[~idx] = 0.
            
            c_offset[i,j,:] = np.sum(w*dc, axis=0) / np.sum(w, axis=0)
    
    # Extend c_offset to uniform bright magnitude
    M = mags[1,:,b1]
    
    # Loop over [Fe/H]
    for i in [0,2]:
        M_i = mags[i,:,b1]
        
        # Find first point that is not included at bright end
        idx_0 = np.sum(M < np.nanmin(M_i)) + 5
        
        M_0 = M[idx_0]
        c_offset_0 = c_offset[i,idx_0]
        
        dM = M[idx_0+10] - M[idx_0]
        dc_offset = c_offset[i,idx_0+10,:] - c_offset[i,idx_0,:]
        
        c_offset[i,:idx_0,:] = c_offset_0 + np.einsum('i,j->ij', (M[:idx_0] - M_0)/dM, dc_offset)
        
        for k in xrange(colors.shape[2]):
            # Find first point that is not included at faint end
            idx_0 = np.max(np.where(np.isfinite(c_offset[i,:,k]))[0]) - 50
            
            M_0 = M[idx_0]
            print i, k, M_0
            c_offset_0 = c_offset[i,idx_0,k]
            
            dM = M[idx_0-10] - M[idx_0]
            dc_offset = c_offset[i,idx_0-10,k] - c_offset[i,idx_0,k]
            
            c_offset[i,idx_0:,k] = c_offset_0 + (M[idx_0:] - M_0)/dM * dc_offset
    
    # Smooth color offsets
    for i in xrange(c_offset.shape[0]):
        for k in xrange(c_offset.shape[2]):
            idx = np.isfinite(c_offset[i,:,k])
            c_offset[i,idx,k] = gaussian_filter1d(c_offset[i,idx,k], sigma=5., mode='nearest')
    
    
    #for j in xrange(c_offset.shape[1]):
    #    print mags[1,j,b1], c_offset[-1,j,0]
    
    
    #
    # Generate models from offsets
    #
    
    colors_new = np.empty(colors.shape, dtype='f8')
    
    for i in xrange(colors.shape[0]):
        colors_new[i, :, :] = colors[1, :, :]
    
    colors_new += c_offset
    
    
    
    #mags[1,:,0] = gaussian_filter1d(mags[1,:,0], sigma=5., mode='nearest')
    
    mags_new = np.empty(mags.shape, dtype='f8')
    for k in xrange(mags.shape[0]):
        mags_new[:,:,0] = mags[1,:,0]
    
    for k in xrange(mags.shape[2]-1):
        mags_new[:,:,k+1] = mags[1,:,0] - colors_new[:,:,k]
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(mags[0,:,1], c_offset[0, :, 1], 'b-')
    ax.plot(mags[0,:,1], c_offset[2, :, 1], 'g-')
    
    plt.show()
    '''
    
    #
    # Interpolate over [Fe/H]
    #
    
    mags_new = interp_linear_extensive(mags_new, n=1, axis=0)
    FeH_new = interp_linear_extensive(props['FeH'], n=1)
    #FeH_new = props['FeH']
    
    idx = (FeH_new >= -3.0) & (FeH_new <= 0.5)
    FeH_new = FeH_new[idx]
    mags_new = mags_new[idx,:,:]
    
    #
    # Plot new fiducials
    #
    
    if plot:
        b1 = 0
        
        bands = 'grizy'
        
        mplib.rc('text', usetex=True)
        
        fig = plt.figure(figsize=(10,6), dpi=100)
        axgrid = Grid(fig, 111,
                      nrows_ncols=(1,4),
                      axes_pad=0.0,
                      add_all=False,
                      label_mode='L')
        
        F_0, F_1 = np.min(FeH_new), np.max(FeH_new)
        DF = F_1 - F_0
        
        jet = plt.get_cmap('jet')
        c_norm = mplib.colors.Normalize(vmin=F_0, vmax=F_1)
        sm = mplib.cm.ScalarMappable(norm=c_norm, cmap=jet)
        sm._A = []
        
        for b2,b_name in enumerate(bands):
            if b2 == 0:
                continue
            
            ax = axgrid[b2-1]
            fig.add_axes(ax)
            
            for i,(p,m) in enumerate(zip(props, mags)):
                c = sm.to_rgba(p['FeH'])
                x = m[:,b2-1]-m[:,b2]
                idx = np.isfinite(x)
                
                ax.plot(x[idx], m[idx,b1], c=c)
                #ax.scatter(x[idx], m[idx,b1])
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            for i,FeH in enumerate(FeH_new):
                #x = colors_new[i,:,b2-1]
                x = mags_new[i,:,b2-1] - mags_new[i,:,b2]
                idx = np.isfinite(x)
                
                c = sm.to_rgba(FeH)
                #ax.plot(x[idx], mags_new[i,idx,b1], c=c, ls='--')
                ax.scatter(x[idx], mags_new[i,idx,b1], c=c, edgecolor='none')
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            ax.set_xlabel(r'$%s - %s$' % (bands[b2-1], bands[b2]), fontsize=16)
            ax.set_ylabel(r'$%s$' % (bands[b1]), fontsize=16)
            
            if b2 == 4:
                ylim = ax.get_ylim()
                ax.set_ylim(ylim[1], ylim[0])
        
        fig.suptitle(r'$\mathrm{GC \ Fiducials}$', fontsize=18)
        fig.subplots_adjust(left=0.08, right=0.85, bottom=0.10, top=0.90)
        cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])
        cbar = fig.colorbar(sm, orientation='vertical', cax=cax)
        cax.set_ylabel(r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', fontsize=14)
        
        plt.show()
    
    
    return mags_new, FeH_new


def fit_g_offset(plot=False):
    # Load GC fiducials and clip to main sequence
    props, mags = load_fiducials(selection=[7078, 5904, 6838, 6791])
    
    for i,m in enumerate(mags):
        mags[i] -= props['DM'][i]
        
        idx = (m[:,1] < 4.)
        mags[i][idx,:] = np.nan
    
    # Determine gr vs. ri in each GC
    gc_gr = []
    gc_ri = []
    gc_gr_spl = []
    
    for m in mags:
        idx = np.isfinite(m)
        idx = idx[:,0] & idx[:,1] & idx[:,2]
        
        gc_gr.append(m[idx,0] - m[idx,1])
        gc_ri.append(m[idx,1] - m[idx,2])
    
    gc_gr_spl = [dp.NaturalCubicSpline(ri.tolist(), gr.tolist()) for gr,ri in zip(gc_gr, gc_ri)]
    
    # Load main-sequence locus fits (from NGP)
    fname = os.path.expanduser('~/projects/stellar-locus/PS1_2MASS_locus_points_2.txt')
    ngp_loci = read_ascii_table(fname)
    
    ngp_gr_spl = dp.NaturalCubicSpline(ngp_loci[:,1].tolist(), ngp_loci[:,0].tolist())
    
    # Offset in gr as function of ri at each metallicity
    ri_range = np.linspace(np.min(ngp_loci[:,1]), np.max(ngp_loci[:,1]), 100)
    
    ngp_gr_eval = np.array(ngp_gr_spl(ri_range.tolist()))
    
    gc_gr_eval = [np.array(spl(ri_range.tolist())) for spl in gc_gr_spl]
    gr_offset = [gr - ngp_gr_eval for gr in gc_gr_eval]
    
    # Smooth gr offset
    
    gr_offset_sm_1 = [gaussian_filter1d(gr.astype('f8'), 1.) for gr in gr_offset]
    gr_offset_sm_2 = [gaussian_filter1d(gr.astype('f8'), 5.) for gr in gr_offset]
    a = 0.5 * (1. + scipy.special.erf(0.5*np.sqrt(np.pi)*(ri_range - 0.25)/0.10))
    
    gr_offset_sm = a*gr_offset_sm_1 + (1.-a)*gr_offset_sm_2
    
    # Continue gr offset of high-metallicity cluster indefinitely at same value
    y = gr_offset_sm[-1]
    i_min = np.argmin(y[ri_range > 0.4]) + np.sum(ri_range <= 0.4)
    y[i_min:] = y[i_min]
    gr_offset_sm[-1] = y
    
    # Limit gr offset of low-metallicity cluster to be less than 0.015
    y_0 = -y[i_min]
    y = gr_offset_sm[0]
    #y_0 = 0.015
    dy = 0.03
    y_p = y_0 - y
    y_p = dy * np.log(np.exp(y_0) + np.exp(y_p/dy))
    y = y_0 - y_p
    gr_offset_sm[0] = y
    
    # Sample gr offset more finely in [Fe/H]
    gr_offset_sm = np.array(gr_offset_sm)
    
    idx = [0,-1]
    dgr_interp = interp_linear_extensive(gr_offset_sm[idx,:], n=2, axis=0)
    FeH_interp = interp_linear_extensive(props['FeH'][idx], n=2)
    
    
    if plot:
        # Plot gr offset
        mplib.rc('text', usetex=True)
        
        F_0, F_1 = np.min(FeH_interp), np.max(FeH_interp)
        DF = F_1 - F_0
        
        jet = plt.get_cmap('jet')
        c_norm = mplib.colors.Normalize(vmin=F_0, vmax=F_1)
        sm = mplib.cm.ScalarMappable(norm=c_norm, cmap=jet)
        sm._A = []
        
        fig = plt.figure(figsize=(9,4), dpi=100)
        
        ax = fig.add_subplot(1,2,1)
        
        ax.plot(ngp_gr_eval, ri_range, 'k-', alpha=0.5)
        
        for gr,p in zip(gc_gr_eval, props):
            c = sm.to_rgba(p['FeH'])
            ax.plot(gr, ri_range, c=c, alpha=0.5)
        
        ax = fig.add_subplot(1,2,2)
        
        ax.axhline(0., c='k', alpha=0.5)
        
        for dgr,dgr_sm,p in zip(gr_offset, gr_offset_sm, props):
            c = sm.to_rgba(p['FeH'])
            ax.plot(ri_range, dgr, c=c, alpha=0.75)
            ax.plot(ri_range, dgr_sm, c=c, ls='--', alpha=0.75)
            #print dgr_sm
        
        for dgr,FeH in zip(dgr_interp, FeH_interp):
            c = sm.to_rgba(FeH)
            ax.plot(ri_range, dgr, c=c, alpha=1.)
        
        
        #
        # Color-color diagram
        #
        
        fig = plt.figure(figsize=(10,10), dpi=100)
        axgrid = Grid(fig, 111,
                      nrows_ncols=(3,3),
                      axes_pad=0.0,
                      add_all=False,
                      label_mode='L')
        
        for row in xrange(3):
            i_1 = row+1
            c_1 = [m[:,i_1] - m[:,i_1+1] for m in mags]
            
            for col in xrange(row+1):
                i_2 = col
                c_2 = [m[:,i_2] - m[:,i_2+1] for m in mags]
                
                ax = axgrid[3*row + col]
                fig.add_axes(ax)
                
                for cc_1, cc_2, p in zip(c_1, c_2, props):
                    c = sm.to_rgba(p['FeH'])
                    idx = np.isfinite(cc_1) & np.isfinite(cc_2)
                    ax.plot(cc_2[idx], cc_1[idx], c=c)
                
                ax.plot(ngp_loci[:,col], ngp_loci[:,row+1], 'k-', alpha=0.5)
        
        plt.show()
    
    return FeH_interp, ri_range, dgr_interp



def main():
    fit_GCs(plot=True)
    
    #props, mags = load_fiducials(selection=[7078, 5904, 6838, 6791])
    #plot_fiducials(props, mags)
    
    #fit_g_offset(plot=True)
    
    return 0

if __name__ == '__main__':
    main()

