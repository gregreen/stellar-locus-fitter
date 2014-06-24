#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  replace_colors.py
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

from scipy.interpolate import interp1d, UnivariateSpline

import decimalpy as dp

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import argparse, sys


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


def main():
    parser = argparse.ArgumentParser(prog='replace_colors.py',
                       description='Replace colors in stellar template file with new color fit.',
                       add_help=True)
    parser.add_argument('-t', '--templates', type=str, required=True,
                                          help='Stellar templates to use as basis.')
    parser.add_argument('-o', '--output', type=str, required=True,
                                          help='Output filename.')
    parser.add_argument('-c', '--colors', type=str, required=True,
                                          help='File containing new stellar locus.')
    parser.add_argument('-m', '--match', type=int, nargs=2, required=True,
                                          help='Column in original templates and new'
                                               'stellar locus to match, respectively.')
    #parser.add_argument('-app', '--append', action='store_true',
    #                                      help='Append new colors without replacing old ones and new templates.')
    parser.add_argument('-s', '--show', action='store_true',
                                          help='Plot stellar locus and new templates.')
    
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])
    
    col1, col2 = args.match
    
    # Load new stellar locus
    colors = read_ascii_table(args.colors).T
    n_c = colors.shape[0]
    
    # Spline fits to colors
    w = np.ones(colors.shape[1], dtype='f8') / 0.0025
    #spl = [UnivariateSpline(colors[col2], c, k=3, s=0.) for c in colors]
    spl = [dp.NaturalCubicSpline(colors[col2].tolist(), c.tolist()) for c in colors]
    
    # Plot spline fits to colors
    if args.show:
        fig = plt.figure()
        
        c_min, c_max = np.min(colors[col2]), np.max(colors[col2])
        c_match = np.linspace(c_min-0.5, c_max+0.5, 1000)
        
        for i,s in enumerate(spl):
            ax = fig.add_subplot(1, n_c, i+1)
            ax.plot(c_match, s(c_match.tolist()), 'b-')
            ax.scatter(colors[col2], colors[i], c='k', s=10)
        
        plt.show()
    
    # Load old templates
    tmp = read_ascii_table(args.templates)
    MrFeH = tmp[:,:2]
    c_old = tmp[:,2+col1].tolist()
    #print c_old.shape
    #print c_old
    
    templates = np.empty((MrFeH.shape[0], 2+n_c), dtype='f8')
    templates[:,:2] = MrFeH[:,:]
    
    for i,s in enumerate(spl):
        print 'Filling color %d ...' % i
        
        if i == col2:
            templates[:, 2+i] = c_old[:]
            continue
        #elif i in args.keep:
        #    templates[:, 2+i] = c_old_i[i, :]
        
        templates[:, 2+i] = s(c_old)
    
    txt = '\n'.join(['  '.join(['%.4f' % t for t in line]) for line in templates])
    
    f = open(args.output, 'w')
    f.write(txt)
    f.close()
    
    # Plot color-magnitude diagrams of models
    if args.show:
        fig = plt.figure()
        
        for i in xrange(n_c):
            ax = fig.add_subplot(1, n_c, i+1)
            ax.scatter(templates[:,2+i], templates[:,0], c=templates[:,1], s=3, edgecolor='none')
            ax.set_ylim([14., -2.])
        
        plt.show()
    
    return 0

if __name__ == '__main__':
    main()

