#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gui_fit.py
#  
#  Copyright 2014 Gregory M. Green <greg@greg-UX31A>
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

from scipy.ndimage.filters import gaussian_filter

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import argparse, sys
import time

import h5py

import fit_sl


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
# Plotting
#

band_dispname_dict = {'u_SDSS':'u_{\mathrm{SDSS}}',
                      'g_SDSS':'g_{\mathrm{SDSS}}',
                      'r_SDSS':'r_{\mathrm{SDSS}}',
                      'i_SDSS':'i_{\mathrm{SDSS}}',
                      'z_SDSS':'z_{\mathrm{SDSS}}',
                      'g_P1':'g_{\mathrm{P1}}',
                      'r_P1':'r_{\mathrm{P1}}',
                      'i_P1':'i_{\mathrm{P1}}',
                      'z_P1':'z_{\mathrm{P1}}',
                      'y_P1':'y_{\mathrm{P1}}',
                      'g_DES':'g_{\mathrm{DES}}',
                      'r_DES':'r_{\mathrm{DES}}',
                      'i_DES':'i_{\mathrm{DES}}',
                      'z_DES':'z_{\mathrm{DES}}',
                      'Y_DES':'Y_{\mathrm{DES}}',
                      'J':'J',
                      'H':'H',
                      'K_s':'K_{s}'}


class LocusFitter:
    def __init__(self, fig, colors, c=('g', 'r'), lw=2., alpha=0.5):
        self.fig = fig
        self.colors = colors
        self.n_c = colors.shape[1]
        self.c = c
        self.lw = lw
        self.alpha = alpha
        
        self.ax = [[None for j in xrange(self.n_c)] for i in xrange(self.n_c)]
        
        self.r_start = [None for i in xrange(self.n_c)]
        self.r_end = [None for i in xrange(self.n_c)]
        
        self.h_start_drawn = [False for i in xrange(self.n_c)]
        self.h_end_drawn = [False for i in xrange(self.n_c)]
        
        self.v_start_drawn = [False for i in xrange(self.n_c)]
        self.v_end_drawn = [False for i in xrange(self.n_c)]
        
        self.r = None
        
        self.cid = []
    
    def add_axes(self, ax, row, col):
        self.cid.append( ax.figure.canvas.mpl_connect('button_press_event', self) )
        self.ax[row][col] = ax
    
    def _locate_event(self, event):
        for i,row in enumerate(self.ax):
            for j,ax in enumerate(row):
                if event.inaxes == ax:
                    return (i, j)
        return None, None
    
    def _plot_hlines(self, row, y, start=True):
        if start:
            if self.h_start_drawn[row]:
                return
            self.h_start_drawn[row] = True
            c = self.c[0]
        else:
            if self.h_end_drawn[row]:
                return
            self.h_end_drawn[row] = True
            c = self.c[1]
        
        for k in xrange(self.n_c):
            ax = self.ax[row][k]
            
            if ax != None:
                ax.axhline(y, c=c, lw=self.lw, alpha=self.alpha)
    
    def _plot_vlines(self, col, x, start=True):
        if start:
            if self.v_start_drawn[col]:
                return
            self.v_start_drawn[col] = True
            c = self.c[0]
        else:
            if self.v_end_drawn[col]:
                return
            self.v_end_drawn[col] = True
            c = self.c[1]
        
        for k in xrange(self.n_c):
            ax = self.ax[k][col]
            
            if ax != None:
                ax.axvline(x, c=c, lw=self.lw, alpha=self.alpha)
    
    def _plot_sl(self):
        r = self.get_stellar_locus()
        
        if r == None:
            print 'No stellar locus fit present.'
            return
        
        for row in xrange(self.n_c):
            for col in xrange(self.n_c):
                ax = self.ax[row][col]
                
                if ax != None:
                    ax.scatter(r[1:-1,col], r[1:-1,row],
                               c='k', edgecolor='none',
                               s=10, alpha=0.75)
        
        self.fig.canvas.draw()
    
    def __call__(self, event):
        # Determine which subplot was clicked
        i, j = self._locate_event(event)
        
        if i == None:
            return
        
        if self.r != None:
            return
        
        # Get click location
        x = event.xdata
        y = event.ydata
        
        # Set r_start or r_end?
        if (self.r_start[i] == None) or (self.r_start[j] == None):
            if self.r_start[i] == None:
                self.r_start[i] = y
                self._plot_hlines(i, y, start=True)
                self._plot_vlines(i, y, start=True)
            if self.r_start[j] == None:
                self.r_start[j] = x
                self._plot_hlines(j, x, start=True)
                self._plot_vlines(j, x, start=True)
        else:
            if self.r_end[i] == None:
                self.r_end[i] = y
                self._plot_hlines(i, y, start=False)
                self._plot_vlines(i, y, start=False)
            if self.r_end[j] == None:
                self.r_end[j] = x
                self._plot_hlines(j, x, start=False)
                self._plot_vlines(j, x, start=False)
        
        self.fig.canvas.draw()
        
        # Check if done
        for k in xrange(self.n_c):
            if self.r_start[k] == None:
                return
            if self.r_end[k] == None:
                return
        
        #print self.r_start
        #print self.r_end
        
        # Fit stellar locus
        time.sleep(2.)
        print 'Fitting stellar locus ...'
        
        self.r = self._fit_locus()
        
        print 'Stellar locus:'
        
        for rr in self.r:
            print rr
        
        self._plot_sl()
    
    def _fit_locus(self):
        r = self._get_start_end()
        d_max = 0.25 * self._get_width()
        
        r = fit_sl.fit_locus(self.colors, r, d_max,
                             clip=2., n_sigma=2., max_iter=100)
        r = fit_sl.refine_locus(self.colors, r, d_max, clip=2., n_iter=5)
        
        return r
    
    def _get_width(self):
        w = []
        
        for i in xrange(self.n_c):
            x_0, x_1 = np.percentile(self.colors[:, i], [2.5, 97.5])
            w.append(x_1 - x_0)
        
        return np.array(w)
    
    def _get_start_end(self):
        r = np.empty((2, self.n_c), dtype='f8')
        
        r[0,:] = np.array(self.r_start)
        r[1,:] = np.array(self.r_end)
        
        return r
    
    def get_stellar_locus(self):
        return self.r


#
# I/O
#

def read_colors(fname, bands):
    # Read h5 file
    f = h5py.File(fname, 'r')
    data = f['phot'][:]
    f.close()
    
    # De-redden
    errs = data['errs'][:]
    mags = dereddened_mags(data['mags'], data['EBV'], bands)
    
    # Filter out bad photometry
    idx = (
              np.all(np.isfinite(mags), axis=1)
            & np.all(np.isfinite(errs), axis=1)
            & np.all(errs < 0.5, axis=1)
          )
    
    # Filter out large reddenings
    idx &= (data['EBV'] < 0.1)
    
    mags = mags[idx]
    errs = errs[idx]
    
    # Resample photometry from errors
    m = []
    
    for i in xrange(10):
        dm = errs * np.random.normal(size=errs.shape, scale=1.)
        m.append(mags + dm)
    
    mags = np.concatenate(m, axis=0)
    colors = -np.diff(mags, axis=1)
    
    return colors, np.median(data['EBV'])


def main():
    parser = argparse.ArgumentParser(prog='gui_fit.py',
                                     description='Fit arbitrary multidimensional color-color diagrams.',
                                     add_help=True)
    parser.add_argument('photometry', type=str,
                        help="HDF5 file with photometry.\n"
                             "The photometry should be stored in a structured dataset\n"
                             "named 'phot', with three fields, 'mags', 'errs', and 'EBV'.")
    parser.add_argument('bands', type=str, nargs='+',
                        help="Names of photometric passbands, in same order as they\n"
                             "appear in 'mags' and 'errs' fields in photometry.\n"
                             "Ex.: 'g_P1 r_P1 i_P1 z_P1 y_P1' for the Pan-STARRS1 bands.")
    parser.add_argument('--plt-fn', '-plt', type=str, default=None, help='Plot filename.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename for stellar locus points.')
    
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])
    
    # Load photometry
    colors, EBV = read_colors(args.photometry, args.bands)
    n_stars, n_c = colors.shape
    
    
    #
    # Plot color-color diagram
    #
    
    # Determine display limits
    lim = []
    
    for i in xrange(n_c):
        x_0, x_1 = np.percentile(colors[:, i], [2.5, 97.5])
        w = x_1 - x_0
        lim.append([x_0 - 0.2*w, x_1 + 0.2*w])
    
    # Look up reddening vector
    R_c = get_reddening_vector(args.bands)
    R_c = -np.diff(R_c)
    
    # Set matplotlib style attributes
    mplib.rc('text', usetex=True)
    mplib.rc('xtick.major', size=6)
    mplib.rc('xtick.minor', size=4)
    mplib.rc('ytick.major', size=6)
    mplib.rc('ytick.minor', size=4)
    mplib.rc('xtick', direction='in')
    mplib.rc('ytick', direction='in')
    mplib.rc('axes', grid=False)
    
    # Set up figure
    fig = plt.figure(figsize=(8,8), dpi=150)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(n_c-1,n_c-1),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    # Set up stellar locus fitter
    fitter = LocusFitter(fig, colors)
    
    # Grid of axes
    for row in xrange(6):
        color_y = colors[:,row+1]
        
        for col in xrange(row+1):
            color_x = colors[:,col]
            
            ax = axgrid[(n_c-1)*row + col]
            fig.add_axes(ax)
            
            fitter.add_axes(ax, row+1, col)
            
            xlim = lim[col]
            ylim = lim[row+1]
            w = xlim[1] - xlim[0]
            h = ylim[1] - ylim[0]
            
            # Density plot
            rho, tmp, tmp = np.histogram2d(color_x, color_y,
                                           range=[xlim, ylim],
                                           bins=200)
            rho = gaussian_filter(rho, 2.)
            img = np.sqrt(rho)
            
            ax.imshow(img.T, origin='lower', aspect='auto', interpolation='none',
                             extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
            
            # Reddening vector
            x_c = 0.5*(xlim[0] + xlim[1])
            y_c = 0.5*(ylim[0] + ylim[1])
            
            R_x = R_c[col]
            R_y = R_c[row+1]
            
            a = 0.5
            ax.arrow(x_c-0.5*a*R_x, y_c-0.5*a*R_y,
                     0.5*a*R_x, 0.5*a*R_y,
                     head_length=0.02, head_width=0.02,
                     fc='r', ec='r', alpha=0.75)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    
    # Format x-axes
    band_names = [band_dispname_dict[b] for b in args.bands]
    
    for i,b in enumerate(zip(band_names[:-2], band_names[1:-1])):
        color_label = r'$%s - %s$' % (b[0], b[1])
        ax = axgrid[30+i]
        ax.set_xlabel(color_label, fontsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Format y-axes
    for i,b in enumerate(zip(band_names[1:-1], band_names[2:])):
        color_label = r'$%s - %s$' % (b[0], b[1])
        ax = axgrid[6*i]
        ax.set_ylabel(color_label, fontsize=14)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    fig.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.98)
    
    txt = r'$\mathrm{\# \ of \ stars:} \ %d$' % n_stars
    txt += '\n'
    txt += r'$\left< \mathrm{E} \left( B - V \right) \right> = %.3f$' % EBV
    fig.text(0.98, 0.98, txt,
             fontsize=16, ha='right', va='top', multialignment='center')
    
    plt.show()
    
    # Write stellar locus to ASCII file
    if args.output != None:
        r = fitter.get_stellar_locus()
        txt = '\n'.join(['  '.join(['%.4f' % c for c in pt]) for pt in r])
        
        f = open(args.output, 'w')
        f.write(txt)
        f.close()    
    
    return 0

if __name__ == '__main__':
    main()

