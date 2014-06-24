#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gen_templates.py
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

import decimalpy as dp

import scipy.interpolate
import scipy.special

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import itertools

import os

from ps import sdsspstransformall
from cubicspline import CubicSpline

import gc_fiducials


def cartesian_product(arrays):
    return np.array([x for x in itertools.product(*arrays)])


def read_ascii_table(fname):
    '''
    Read an ASCII table, returning a numpy array.
    
    Input:
        fname  Filename of ASCII table
    
    Output:
        numpy float array with shape (n_rows, n_columns)
    '''
    
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


def ps2sdss_colors(ps_colors):
    '''
    Transform PS1 colors to SDSS colors, additionally returning the
    offset in the r-band magnitudes (SDSS - PS1).
    
    Input:
        ps_colors    (n_stars, 4)
    
    Output:
        sdss_colors  (n_stars, 4)
        
    '''
    
    ps_mags = np.empty((ps_colors.shape[0], 5), dtype='f8')
    ps_mags[:,0] = 0.
    ps_mags[:,1:] = -np.cumsum(ps_colors, axis=1)
    
    sdss_mags = sdsspstransformall(ps_mags)
    sdss_colors = -np.diff(sdss_mags)
    
    Delta_r = sdss_mags[:,2] - ps_mags[:,1]
    
    return sdss_colors, Delta_r


def Mr_relation(gi_sdss, FeH):
    '''
    Calculate absolute r magnitude for a given g-i color (in SDSS)
    and [Fe/H], using the parallax relation from Ivezić+ (2008; See
    Eqs. A1-A4).
    '''
    
    #Mr_0 = -2.85 + 6.29 * gi_sdss - 2.30 * gi_sdss**2
    Mr_0 = -1.93 + 4.39 * gi_sdss - 1.73 * gi_sdss**2 + 0.452 * gi_sdss**3
    Delta_Mr = 4.50 - 1.11 * FeH - 0.18 * FeH**2
    
    return Mr_0 + Delta_Mr


def ps_MS_mags(ps_loci, FeH_range=np.linspace(-3.0, 0.5, 8)):
    # Transform PS1 locus points to SDSS colors
    sdss_loci, Delta_r = ps2sdss_colors(ps_loci[:,:4])
    
    # Get SDSS Mr from parallax relation
    Mr_sdss = []
    gi = sdss_loci[:,1] + sdss_loci[:,2]
    
    for FeH in FeH_range:
        Mr_sdss.append(Mr_relation(gi, FeH))
    
    # Transform back into PS1 bands
    ps_mags = []
    
    for Mr in Mr_sdss:
        m = np.empty((ps_loci.shape[0], 8), dtype='f8')
        m[:,0] = 0.
        m[:,1:] = -np.cumsum(ps_loci, axis=1)
        
        dm = np.empty(m.shape[0], dtype='f8')
        dm[:] = m[:,1]
        
        for i in xrange(8):
            m[:,i] += -dm + Mr - Delta_r
        
        ps_mags.append(m)
    
    return np.array(ps_mags)


def ps_MS_mags_corr2(ps_loci, FeH_range=np.linspace(-3.0, 0.5, 8)):
    # Transform PS1 locus points to SDSS colors
    sdss_loci, Delta_r = ps2sdss_colors(ps_loci[:,:4])
    
    # Get SDSS Mr from parallax relation
    Mr_sdss = []
    gi = sdss_loci[:,1] + sdss_loci[:,2]
    
    for FeH in FeH_range:
        Mr_sdss.append(Mr_relation(gi, FeH))
    
    # Transform back into PS1 bands
    ps_mags = []
    
    for Mr in Mr_sdss:
        m = np.empty((ps_loci.shape[0], 8), dtype='f8')
        m[:,0] = 0.
        m[:,1:] = -np.cumsum(ps_loci, axis=1)
        
        dm = np.empty(m.shape[0], dtype='f8')
        dm[:] = m[:,1]
        
        for i in xrange(8):
            m[:,i] += -dm + Mr - Delta_r
        
        ps_mags.append(m)
    
    # Calculate g-r offset as function of r-i and [Fe/H]
    print 'Calculating g-r offset as function of r-i and [Fe/H]...'
    FeH_dgr, ri_dgr, dgr = gc_fiducials.fit_g_offset()
    dgr_interp = scipy.interpolate.interp2d(FeH_dgr, ri_dgr, dgr.T)
    
    # Apply color correction for metallicity
    print 'Applying g-r color correction for metallicity...'
    ps_mags_corr = []
    
    for m,FeH in zip(ps_mags, FeH_range):
        # Apply color correction for metallicity
        ri = m[:,1] - m[:,2]
        dgr_tmp = dgr_interp(FeH, ri)
        
        m_corr = np.empty(m.shape, dtype='f8')
        m_corr[:] = m[:]
        
        m_corr[:,0] -= dgr_tmp[:].flatten()
        
        ps_mags_corr.append(m_corr)
    
    return np.array(ps_mags_corr)


def ps_MS_mags_corr(ps_loci, FeH_range=np.linspace(-3.0, 0.5, 8)):
    # Calculate g-r offset as function of r-i and [Fe/H]
    print 'Calculating g-r offset as function of r-i and [Fe/H]...'
    FeH_dgr, ri_dgr, dgr = gc_fiducials.fit_g_offset()
    dgr_interp = scipy.interpolate.interp2d(FeH_dgr, ri_dgr, dgr.T)
    
    # Apply color correction for metallicity
    print 'Applying g-r color correction for metallicity...'
    ps_loci_corr = []
    
    for FeH in FeH_range:
        # Apply color correction for metallicity
        ri = ps_loci[:,1]
        dgr_tmp = dgr_interp(FeH, ri)
        
        #print dgr_tmp
        #print dgr_tmp.shape
        
        tmp = np.empty(ps_loci.shape, dtype='f8')
        tmp[:] = ps_loci[:]
        tmp[:,0] += dgr_tmp[:].flatten()
        
        ps_loci_corr.append(tmp)
    
    ps_loci_corr = np.array(ps_loci_corr)
    
    # Get SDSS Mr from SDSS parallax relation
    print 'Calculating Mr from SDSS parallax relation...'
    Mr_sdss = []
    
    for ps_c, FeH in zip(ps_loci_corr, FeH_range):
        # Transform PS1 locus points to SDSS colors
        sdss_loci, Delta_r = ps2sdss_colors(ps_c[:,:4])
        
        gi = sdss_loci[:,1] + sdss_loci[:,2]
        Mr_sdss.append(Mr_relation(gi, FeH))
    
    # Transform back into PS1 bands
    print 'Transforming back to PS1 bands...'
    ps_mags = []
    
    for ps_c, Mr in zip(ps_loci_corr, Mr_sdss):
        m = np.empty((ps_c.shape[0], 8), dtype='f8')
        m[:,0] = 0.
        m[:,1:] = -np.cumsum(ps_c, axis=1)
        
        dm = np.empty(m.shape[0], dtype='f8')
        dm[:] = m[:,1]
        
        for i in xrange(8):
            m[:,i] += -dm + Mr - Delta_r
        
        ps_mags.append(m)
    
    return np.array(ps_mags)


def gen_MS_models(FeH_range=np.linspace(-3.0, 0.5, 8), Mg_range=np.linspace(2., 20., 431)):
    # Load color-color locus points
    fname = os.path.expanduser('~/projects/stellar-locus/PS1_2MASS_locus_points_2.txt')
    ps_loci = read_ascii_table(fname)
    
    ps_MS = ps_MS_mags_corr2(ps_loci, FeH_range=FeH_range)
    
    # Interpolate onto regular Mg grid
    
    ps_colors = np.empty((Mg_range.size, FeH_range.size, 7), dtype='f8')
    
    for j,FeH in enumerate(FeH_range):
        print 'Calculating spline for [Fe/H] = %.3f (%d of %d) ...' % (FeH, j+1, len(FeH_range))
        for k in xrange(7):
            #spl = CubicSpline(ps_MS[j,:,1], ps_MS[j,:,k]-ps_MS[j,:,k+1])
            #ps_colors[:,j,k] = spl(Mr_range)
            spl = dp.NaturalCubicSpline(ps_MS[j,:,0].tolist(), (ps_MS[j,:,k]-ps_MS[j,:,k+1]).tolist())
            ps_colors[:,j,k] = spl(Mg_range.tolist())
    
    #
    # Convert to magnitudes
    #
    
    ps_mags = np.empty((Mg_range.size, FeH_range.size, 8), dtype='f8')
    
    for j in xrange(FeH_range.size):
        ps_mags[:,j,0] = Mg_range[:]
    
    #ps_mags[:,:,0] = ps_mags[:,:,1] + ps_colors[:,:,0]
    
    for k in xrange(1,8):
        ps_mags[:,:,k] = ps_mags[:,:,k-1] - ps_colors[:,:,k-1]
    
    '''
    # Plot results
    fig = plt.figure()
    
    jet = plt.get_cmap('jet')
    c_norm = mplib.colors.Normalize(vmin=FeH_range[0], vmax=FeH_range[-1])
    sm = mplib.cm.ScalarMappable(norm=c_norm, cmap=jet)
    sm._A = []
    
    ps_c_plt = -np.diff(ps_MS, axis=2)
    
    for k in xrange(7):
        ax = fig.add_subplot(1,7,k+1)
        
        for j,FeH in enumerate(FeH_range):
            c = sm.to_rgba(FeH)
            
            ax.plot(ps_colors[:,j,k], Mr_range, 'b-', lw=2, c=c)# alpha=0.5)
            #ax.scatter(ps_c_plt[:,:,k], ps_MS[:,:,1])
        
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[::-1])
    
    plt.show()
    '''
    
    ps_mags = np.swapaxes(ps_mags, 0, 1)
    
    return ps_mags, FeH_range, Mg_range


def stitch_ms_gc():
    print 'Generating GC models...'
    gc_mags, gc_FeH = gc_fiducials.fit_GCs()
    
    print 'Generating MS models...'
    Mg_range = gc_mags[0,:,0]
    idx = np.isfinite(Mg_range)
    
    dg = 0.05
    Mg_range_extension = np.arange(np.nanmax(Mg_range)+dg, 20.+0.1*dg, dg)
    
    Mg_range = np.hstack([Mg_range[idx], Mg_range_extension])
    
    '''
    dg = np.diff(Mg_range)
    idx = np.isfinite(dg)
    dg = dg[idx][0]
    
    Mg_range = np.arange(Mg_range[0], 20.+0.01*dg, dg)
    '''
    
    print Mg_range
    
    ms_mags, ms_FeH, ms_Mg = gen_MS_models(FeH_range=gc_FeH,
                                           Mg_range=Mg_range)
    
    
    print 'Stitching together models...'
    shape = list(ms_mags.shape)
    shape[2] = 5
    mags = np.empty(shape, dtype='f8')
    mags[:] = np.nan
    n1, n2, n3 = gc_mags.shape
    mags[:n1,:n2,:n3] = gc_mags[:,:,:]
    
    idx = (Mg_range > 1.)
    idx.shape = (1, idx.shape[0], 1)
    idx = np.repeat(idx, 5, axis=2)
    idx = np.repeat(idx, n1, axis=0)
    idx &= ~np.isfinite(mags)
    mags[idx] = 0.
    
    Mg_0 = 5.0 #5.0
    dMg_0 = 0.8 #0.55
    
    a = 0.5 * (1. + scipy.special.erf(0.5*np.sqrt(np.pi)*(Mg_range - Mg_0)/dMg_0))
    
    Mg_1 = 4.5 #4.3
    dMg_1 = 0.35 #0.25
    a[Mg_range < Mg_1] *= np.exp(-(Mg_range[Mg_range < Mg_1] - Mg_1)**2 / (0.5 * dMg_1)**2.)
    
    #Mg_2 = 6.1
    #dMg_2 = 0.1
    #a[Mg_range > Mg_2] *= np.exp(-(Mg_range[Mg_range > Mg_2] - Mg_2)**2 / (0.5 * dMg_2)**2.)
    
    for aa,mm,mgc in zip(a, Mg_range, gc_mags[1,:,0]):
        print mm, mgc, aa
    
    a.shape = (1, a.size, 1)
    a = np.repeat(a, 5, axis=2)
    a = np.repeat(a, gc_FeH.size, axis=0)
    a[idx] = 1.
    
    mags[:] = 0.
    mags[:n1,:n2,:n3] = (1. - a[:n1,:n2,:n3]) * gc_mags[:,:,:]
    mags[idx] = 0.
    
    idx = (Mg_range <= 1.)
    idx.shape = (1, idx.shape[0], 1)
    idx = np.repeat(idx, 5, axis=2)
    idx = np.repeat(idx, n1, axis=0)
    idx &= ~np.isfinite(mags)
    mags[idx] = 0.
    
    mags[:] += a * ms_mags[:,:,:5]
    mags[idx] = np.nan
    
    #
    # Interpolate onto regular grid in (Mr, [Fe/H])
    #
    
    #idx = np.all(np.all(np.isfinite(mags), axis=2), axis=0)
    #mags = mags[:,idx,:]
    
    Mr_old = mags[:,:,1].flatten()
    
    FeH_old = np.empty(gc_FeH.shape, dtype='f8')
    FeH_old[:] = gc_FeH[:]
    FeH_old.shape = (FeH_old.shape[0], 1)
    FeH_old = np.repeat(FeH_old, mags.shape[1], axis=1)
    FeH_old.shape = (FeH_old.size,)
    
    Mr_new = np.arange(-2., 18.01, 0.05)
    FeH_new = np.arange(-2.35, 0.4001, 0.05)
    mags_interp = np.empty((FeH_new.size, Mr_new.size, 5), dtype='f8')
    
    FeH_new, Mr_new = cartesian_product([FeH_new, Mr_new]).T
    mags_interp[:,:,1] = np.reshape(Mr_new, (mags_interp.shape[0], mags_interp.shape[1]))
    
    x_old = np.array([FeH_old, Mr_old]).T
    x_new = np.array([FeH_new, Mr_new]).T
    
    x_old += 0.001*(np.random.random(x_old.shape)-0.5)
    
    for k in [0,2,3,4]:
        shape = mags_interp.shape[:2]
        tmp = scipy.interpolate.griddata(x_old, mags[:,:,k].flatten(), x_new, method='linear')
        mags_interp[:,:,k] = np.reshape(tmp, shape)
        #tck = scipy.interpolate.bisplrep(FeH_old, Mr_old, mags[:,:,k].flatten(), s=0.)
        #print scipy.interpolate.bisplev([0.], [0.], tck)
        #mags_interp[:,:,k] = scipy.interpolate.bisplev(FeH_new, Mr_new, tck)
    
    
    #
    # Output grid to ASCII
    #
    
    colors_interp = -np.diff(mags_interp, axis=2)
    colors_interp.shape = (FeH_new.size, colors_interp.shape[2])
    
    lines = ['# Mr    FeH   gr     ri     iz     zy', '#']
    lines += ['%.2f  %.2f  %.4f  %.4f  %.4f  %.4f' % (Mr, FeH, c[0], c[1], c[2], c[3])
              for Mr,FeH,c in zip(Mr_new, FeH_new, colors_interp)]
    txt = '\n'.join(lines)
    
    f = open('colors.dat', 'w')
    f.write(txt)
    f.close()
    
    
    #
    # Plot both sets of templates
    #
    
    b1 = 0
    
    bands = 'grizy'
    
    mplib.rc('text', usetex=True)
    
    fig = plt.figure(figsize=(10,6), dpi=100)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(1,4),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    F_0, F_1 = np.min(ms_FeH), np.max(ms_FeH)
    DF = F_1 - F_0
    
    jet = plt.get_cmap('jet')
    c_norm = mplib.colors.Normalize(vmin=F_0, vmax=F_1)
    sm = mplib.cm.ScalarMappable(norm=c_norm, cmap=jet)
    sm._A = []
    
    #ax_idx = 1
    
    for b2 in xrange(1, len(bands)):
        #if b2 == b1:
        #    continue
        
        ax = axgrid[b2-1]
        #ax_idx += 1
        fig.add_axes(ax)
        
        for i,FeH in enumerate(gc_FeH):
            x = gc_mags[i,:,b2-1] - gc_mags[i,:,b2]
            idx = np.isfinite(x)
            
            c = sm.to_rgba(FeH)
            ax.scatter(x[idx], gc_mags[i,idx,b1],
                       c=c, edgecolor='none', s=2)
        
        for i,FeH in enumerate(ms_FeH):
            x = ms_mags[i,:,b2-1] - ms_mags[i,:,b2]
            idx = np.isfinite(x)
            
            c = sm.to_rgba(FeH)
            ax.scatter(x[idx], ms_mags[i,idx,b1],
                       c=c, edgecolor='none', s=2)
        
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
    
    
    #
    # Plot stitched templates
    #
    
    b1 = 0
    
    bands = 'grizy'
    
    mplib.rc('text', usetex=True)
    
    fig = plt.figure(figsize=(10,6), dpi=100)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(1,4),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    for b2 in xrange(1, len(bands)):
        ax = axgrid[b2-1]
        fig.add_axes(ax)
        
        for i,FeH in enumerate(ms_FeH):
            x = mags[i,:,b2-1] - mags[i,:,b2]
            idx = np.isfinite(x)
            
            c = sm.to_rgba(FeH)
            ax.scatter(x[idx], mags[i,idx,b1],
                       c=c, edgecolor='none', s=2)
        
        ax.set_xlabel(r'$%s - %s$' % (bands[b2-1], bands[b2]), fontsize=16)
        ax.set_ylabel(r'$%s$' % (bands[b1]), fontsize=16)
        
        if b2 == 4:
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[1], ylim[0])
    
    fig.suptitle(r'$\mathrm{Stitched \ Fiducials}$', fontsize=18)
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.10, top=0.90)
    cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])
    cbar = fig.colorbar(sm, orientation='vertical', cax=cax)
    cax.set_ylabel(r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', fontsize=14)
    
    fig.savefig('stitched-color-magnitude.png', dpi=100)
    
    
    #
    # Plot interpolated templates
    #
    
    b1 = 1
    
    bands = 'grizy'
    
    mplib.rc('text', usetex=True)
    
    fig = plt.figure(figsize=(10,6), dpi=100)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(1,4),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    for b2 in xrange(1, len(bands)):
        ax = axgrid[b2-1]
        fig.add_axes(ax)
        
        for i,FeH in enumerate(np.unique(FeH_new)):
            x = mags_interp[i,:,b2-1] - mags_interp[i,:,b2]
            idx = np.isfinite(x)
            
            c = sm.to_rgba(FeH)
            ax.scatter(x[idx], mags_interp[i,idx,b1],
                       c=c, edgecolor='none', s=2)
        
        ax.set_xlabel(r'$%s - %s$' % (bands[b2-1], bands[b2]), fontsize=16)
        ax.set_ylabel(r'$%s$' % (bands[b1]), fontsize=16)
        
        if b2 == 4:
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[1], ylim[0])
    
    fig.suptitle(r'$\mathrm{Interpolated \ Fiducials}$', fontsize=18)
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.10, top=0.90)
    cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])
    cbar = fig.colorbar(sm, orientation='vertical', cax=cax)
    cax.set_ylabel(r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', fontsize=14)
    
    fig.savefig('interpolated-color-magnitude.png', dpi=100)
    
    
    # Color-color
    
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
            
            for cc_1, cc_2, FeH in zip(c_1, c_2, ms_FeH):
                c = sm.to_rgba(FeH)
                idx = np.isfinite(cc_1) & np.isfinite(cc_2)
                ax.plot(cc_2[idx], cc_1[idx], c=c)
    
    fig.savefig('stitched-color-color.png', dpi=100)
    
    plt.show()


def plot_Mr_relation():
    gi = np.linspace(0.0, 4.0, 1000)
    FeH = 0.
    Mr = Mr_relation(gi, FeH)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(gi, Mr)
    
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[::-1])
    
    plt.show()

def test_gen_MS_models():
    FeH_range = np.array([-2.5, -1.0, 0.5])
    ps_mags, FeH_range, Mg_range = gen_MS_models(FeH_range=FeH_range)
    ps_colors = -np.diff(ps_mags, axis=-1)
    
    # Color-magnitude
    fig = plt.figure()
    
    jet = plt.get_cmap('jet')
    c_norm = mplib.colors.Normalize(vmin=FeH_range[0], vmax=FeH_range[-1])
    sm = mplib.cm.ScalarMappable(norm=c_norm, cmap=jet)
    sm._A = []
    
    for k in xrange(7):
        ax = fig.add_subplot(1,7,k+1)
        
        for i,FeH in enumerate(FeH_range):
            c = sm.to_rgba(FeH)
            
            Mr = ps_mags[i,:,1]
            
            ax.plot(ps_colors[i,:,k], Mr, 'b-', lw=2, c=c)# alpha=0.5)
        
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[::-1])
    
    # Color-color
    fig = plt.figure(figsize=(10,10), dpi=100)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(6,6),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    for row in xrange(6):
        i_1 = row+1
        c_1 = [m[:,i_1] - m[:,i_1+1] for m in ps_mags]
        
        for col in xrange(row+1):
            i_2 = col
            c_2 = [m[:,i_2] - m[:,i_2+1] for m in ps_mags]
            
            ax = axgrid[6*row + col]
            fig.add_axes(ax)
            
            for cc_1, cc_2, FeH in zip(c_1, c_2, FeH_range):
                c = sm.to_rgba(FeH)
                idx = np.isfinite(cc_1) & np.isfinite(cc_2)
                ax.plot(cc_2[idx], cc_1[idx], c=c)
    
    plt.show()


def main():
    stitch_ms_gc()
    
    #test_gen_MS_models()
    #plot_Mr_relation()
    
    return 0

if __name__ == '__main__':
    main()

