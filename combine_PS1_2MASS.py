#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  combine_PS1_2MASS.py
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

import matplotlib.pyplot as plt
import matplotlib as mplib
from mpl_toolkits.axes_grid1 import Grid


def main():
    f = open('colors.dat', 'r')
    PS1_lines = [l.lstrip().rstrip().split() for l in f][2:]
    f.close()
    
    f = open('colors_2MASS.dat', 'r')
    tmass_lines = [l.lstrip().rstrip().split()[-3:] for l in f]
    f.close()
    
    lines = [l1 + l2 for l1, l2 in zip(PS1_lines, tmass_lines)]
    txt = '\n'.join(['  '.join([col for col in l]) for l in lines])
    
    f = open('colors_PS1_2MASS_combined.dat', 'w')
    f.write(txt)
    f.close()
    
    #
    # Plot results
    #
    
    tmp = np.array(lines).astype('f8')
    print tmp.shape
    idx = np.arange(tmp.shape[0])
    np.random.shuffle(idx)
    idx = idx[:1000]
    FeH = tmp[idx,1]
    colors = tmp[idx,2:]
    
    fig = plt.figure(figsize=(10,10), dpi=100)
    axgrid = Grid(fig, 111,
                  nrows_ncols=(6,6),
                  axes_pad=0.0,
                  add_all=False,
                  label_mode='L')
    
    jet = plt.get_cmap('jet')
    F_0, F_1 = np.min(FeH), np.max(FeH)
    c_norm = mplib.colors.Normalize(vmin=F_0, vmax=F_1)
    sm = mplib.cm.ScalarMappable(norm=c_norm, cmap=jet)
    sm._A = []
    
    for row in xrange(6):
        #i_1 = row+1
        #c_1 = [m[:,i_1] - m[:,i_1+1] for m in mags]
        c_1 = colors[:,row+1]
        
        for col in xrange(row+1):
            #i_2 = col
            #c_2 = [m[:,i_2] - m[:,i_2+1] for m in mags]
            c_2 = colors[:,col]
            
            ax = axgrid[6*row + col]
            fig.add_axes(ax)
            
            for cc_1, cc_2, F in zip(c_1, c_2, FeH):
                c = sm.to_rgba(F)
                ax.scatter(cc_2, cc_1, c=c, edgecolor='none', s=2)
                #idx = np.isfinite(cc_1) & np.isfinite(cc_2)
                #ax.plot(cc_2[idx], cc_1[idx], c=c)
            
            #ax.plot(ngp_loci[:,col], ngp_loci[:,row+1], 'k-', alpha=0.5)
    
    plt.show()
    
    return 0

if __name__ == '__main__':
    main()

