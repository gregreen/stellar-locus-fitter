#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  color_color.py
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
import scipy, scipy.stats, scipy.special, scipy.ndimage
import h5py
import time

import argparse, sys, os

import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import Grid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, FormatStrFormatter


def get_reddening_vector():
	return np.array([3.172, 2.271, 1.682, 1.322, 1.087, 0.786, 0.508, 0.320])

def dereddened_mags(mags, EBV):
	R = get_reddening_vector()
	if type(EBV) == float:
		R.shape = (1, R.size)
		R = np.repeat(R, len(mags), axis=0)
		return mags - EBV * R
	elif type(EBV) == np.ndarray:
		return mags - np.einsum('i,j->ij', EBV, R)
	else:
		raise TypeError('EBV has unexpected type: %s' % type(EBV))

class KnotLogger:
	def __init__(self, ax, marker='+', c='r', s=4):
		self.ax = ax
		self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
		
		self.x = []
		self.y = []
		
		self.marker = marker
		self.c = c
		self.s = s
	
	def __call__(self, event):
		if event.inaxes != self.ax:
			return
		self.x.append(event.xdata)
		self.y.append(event.ydata)
		if self.marker != None:
			event.inaxes.scatter([event.xdata], [event.ydata],
			                     marker=self.marker, s=self.s, c=self.c)
			self.ax.figure.canvas.draw()
	
	def get_knots(self):
		return self.x, self.y


def main():
	parser = argparse.ArgumentParser(prog='color_color.py',
	                                 description='Plot PS1-2MASS color-color diagrams.',
	                                 add_help=True)
	parser.add_argument('photometry', type=str,
	                    help='Bayestar input file with photometry.')
	parser.add_argument('--output', '-o', type=str, default=None, help='Plot filename.')
	parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
	
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	band_names = ['g', 'r', 'i', 'z', 'y', 'J', 'H', 'K_{s}']
	
	# Load photometry
	f = h5py.File(args.photometry, 'r')
	data = f['phot'][:]
	f.close()
	
	errs = data['errs'][:]
	mags = dereddened_mags(data['mags'], data['EBV'])
	colors = -np.diff(mags, axis=1)
	
	# Compute mask for each color
	idx = []
	for i in xrange(7):
		idx.append( np.isfinite(mags[:,i])
		          & np.isfinite(mags[:,i+1]) )
	
	print colors.shape
	# Compute display limits for each color
	lim = np.empty((7,2), dtype='f8')
	for i in xrange(7):
		lim[i,0], lim[i,1] = np.percentile(colors[idx[i],i], [2., 98.])
	w = np.reshape(np.diff(lim, axis=1), (7,))
	lim[:,0] -= 0.15 * w
	lim[:,1] += 0.15 * w
	
	lim_bounds = np.array([[-0.2, 1.6],
	                       [-0.3, 2.0],
	                       [-0.2, 1.1],
	                       [-0.15, 0.45],
	                       [-1.0, 2.0],
	                       [-1.0, 2.0],
	                       [-1.0, 2.0]])
	for i in xrange(7):
		lim[i,0] = max(lim[i,0], lim_bounds[i,0])
		lim[i,1] = min(lim[i,1], lim_bounds[i,1])
	
	# Compute reddening vector
	R = get_reddening_vector()
	R_c = -np.diff(R)
	print 'Reddening vector:', R_c
	
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
	              nrows_ncols=(6,6),
	              axes_pad=0.0,
	              add_all=False,
	              label_mode='L')
	
	logger = []
	
	# Grid of axes
	for row in xrange(6):
		color_y = colors[:,row+1]
		
		for col in xrange(row+1):
			color_x = colors[:,col]
			idx_xy = idx[col] & idx[row+1] & idx[1]
			
			ax = axgrid[6*row + col]
			fig.add_axes(ax)
			
			xlim = lim[col]
			ylim = lim[row+1]
			w = xlim[1] - xlim[0]
			h = ylim[1] - ylim[0]
			
			logger.append(KnotLogger(ax, s=25))
			
			# Empirical
			alpha = 0.005 * np.power(1000000., 0.40) / np.power(np.sum(idx_xy), 0.40)
			c = colors[idx_xy, 1]
			print np.min(c), np.max(c)
			ax.scatter(color_x[idx_xy], color_y[idx_xy],
			           c=c, vmin=0., vmax=1.5, s=1., alpha=alpha, edgecolor='none')
			
			# Density plot
			#rho, tmp, tmp = np.histogram2d(color_x[idx_xy], color_y[idx_xy],
			#                               range=[xlim, ylim], bins=200)
			
			#rho = scipy.ndimage.filters.gaussian_filter(rho, 2.)
			
			#img = np.sqrt(rho)
			#idx = np.isfinite(img)
			#print np.sum(idx), np.sum(~idx)
			#print np.min(img[idx])
			#if np.sum(~idx) != 0:
			#	img[~idx] = np.min(img[idx])
			
			#ax.imshow(img.T, origin='lower', aspect='auto', interpolation='none',
			#                 extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
			
			# Reddening vector
			x_c = 0.5*(xlim[0] + xlim[1])
			y_c = 0.5*(ylim[0] + ylim[1])
			
			R_x = R_c[col]
			R_y = R_c[row+1]
			
			a = 0.5
			ax.arrow(x_c-0.5*a*R_x, y_c-0.5*a*R_y,
			         0.5*a*R_x, 0.5*a*R_y,
			         fc='r', ec='r', alpha=0.75)
			
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
	
	# Format x-axes
	for i,b in enumerate(zip(band_names[:-2], band_names[1:-1])):
		color_label = r'$%s - %s$' % (b[0], b[1])
		ax = axgrid[30+i]
		ax.set_xlabel(color_label, fontsize=16)
		ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
	
	# Format y-axes
	for i,b in enumerate(zip(band_names[1:-1], band_names[2:])):
		color_label = r'$%s - %s$' % (b[0], b[1])
		ax = axgrid[6*i]
		ax.set_ylabel(color_label, fontsize=16)
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
	
	fig.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.98)
	
	# Information on l.o.s.
	#txt = '$\ell = %.2f^{\circ}$\n' % l
	#txt += '$b = %.2f^{\circ}$\n' % b
	#txt += '$\mathrm{E} \! \left( B \! - \! V \\right) = %.3f$' % (np.median(EBV))
	#fig.text(x_0 + 1.1*w, y_0 + 2.5*h, txt, fontsize=14, ha='left', va='center')
	
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	
	if args.show:
		plt.show()
	
	for i,log in enumerate(logger):
		x, y = log.get_knots()
		if len(x) != 0:
			print ''
			print 'Axis %d:' % (i + 1)
			print x
			print y
	
	return 0

if __name__ == '__main__':
	main()

