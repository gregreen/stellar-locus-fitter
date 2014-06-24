#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  fit_sl.py
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
import matplotlib.pyplot as plt

import geom_utils



def migrate_locus(x, r, k, d_max, clip=2., update_idx=None):
    r_p = np.empty(r.shape, dtype=r.dtype)
    r_p[:] = r[:]
    #r_p[0] = r[0]
    #r_p[-1] = r[-1]
    
    if update_idx == None:
        update_idx = np.arange(r.shape[0]-2)
    
    # Update locus points
    for p in update_idx:
        idx_members = geom_utils.get_locus_members(x, r, k, p, d_max)
        
        if np.sum(idx_members) < 5:
            r_p[p+1] = r[p+1]
            continue
        
        idx_compact, scatter = geom_utils.locus_scatter(x[idx_members], r[p+1], k[p], clip=clip)
        
        if np.sum(idx_compact) < 5:
            r_p[p+1] = r[p+1]
            continue
        
        idx = np.where(idx_members)[0][idx_compact]
        
        r_p[p+1] = np.mean(x[idx], axis=0)
    
    # Update locus vectors
    k_p = geom_utils.get_sl_vectors(r_p[1:-1])
    
    return r_p, k_p


def add_locus_points(x, r, k, d_max, clip=2., n_sigma=3.):
    # Determine width of locus in region of each locus point
    scatter = np.empty(r.shape[0])
    
    for p in xrange(r.shape[0]-2):
        idx_members = geom_utils.get_locus_members(x, r, k, p, d_max)
        
        if np.sum(idx_members) < 5:
            scatter[p+1] = np.max(d_max)
            continue
        
        idx_compact, scatter[p+1] = geom_utils.locus_scatter(x[idx_members], r[p+1], k[p], clip=1.)
    
    scatter[0] = scatter[1]
    scatter[-1] = scatter[-2]
    
    # Add points when distance between locus points exceeds average scatter
    
    scatter = 0.5 * (scatter[:-1] + scatter[1:])
    d = np.sum((r[1:] - r[:-1])**2, axis=1)
    
    r_p = []
    
    for p in xrange(r.shape[0] - 1):
        r_p.append(r[p])
        
        if d[p] > n_sigma * scatter[p]:
            r_p.append(0.5 * (r[p] + r[p+1]))
    
    r_p.append(r[-1])
    r_p = np.array(r_p)
    
    # Update locus vectors
    k_p = geom_utils.get_sl_vectors(r_p[1:-1])
    
    return r_p, k_p


def fit_locus(x, r, d_max, clip=2., n_sigma=3., max_iter=100):
    r = np.array([r[0], r[0], r[1], r[1]])
    k = np.array([r[2] - r[1], r[2] - r[1]])
    
    t_unchanged = 0
    
    for n in xrange(max_iter):
        n_loci = r.shape[0] - 2
        
        print 'Iteration: %d (length = %d)...' % (n, n_loci)
        
        r, k = migrate_locus(x, r, k, d_max, clip=clip)
        r, k = migrate_locus(x, r, k, d_max, clip=clip)
        r, k = add_locus_points(x, r, k, d_max, clip=clip, n_sigma=n_sigma)
        
        if r.shape[0] - 2 == n_loci:
            t_unchanged += 1
            if t_unchanged >= 10:
                break
        else:
            t_unchanged = 0
    
    r, k = migrate_locus(x, r, k, d_max, clip=clip)
    
    return r


def refine_locus(x, r, d_max, clip=2., n_iter=5):
    # Calculate locus vectors
    k = geom_utils.get_sl_vectors(r[1:-1])
    
    # Add new set of locus points
    r, k = add_locus_points(x, r, k, d_max, clip=clip, n_sigma=0.)
    
    # Migrate new locus points
    update_idx = np.arange(0, r.shape[0]-2, 2)
    
    for n in xrange(n_iter):
        n_loci = r.shape[0] - 2
        
        print 'Refining iteration: %d ...' % n
        
        r, k = migrate_locus(x, r, k, d_max, clip=clip, update_idx=update_idx)
    
    return r


#
# Testing functions
#

def fake_locus_mags(pos):
    mags = np.empty((pos.size, 7), dtype='f8')
    
    mags[:, 0] = 1.0*pos
    mags[:, 1] = 0.1*pos - 1.0*pos**2
    mags[:, 2] = 0.2*pos + 0.5*pos**2 - 1.0*pos**3
    mags[:, 3] = 0.3*pos + 0.3*pos**2 + 0.2*pos**3 + 1.0*pos**4
    mags[:, 4] = 0.2*pos - 0.8*pos**2 + 0.7*pos**3 + 0.1*pos**4
    mags[:, 5] = 0.1*pos + 0.2*pos**2 - 0.2*pos**3 - 0.2*pos**4
    mags[:, 6] = 0.1*pos - 0.1*pos**2 - 0.2*pos**3 - 0.1*pos**4
    
    return mags


def gen_fake_locus(n_stars=1000, sigma=0.025, bad_frac=0.1, bad_scale=5.):
    pos = np.random.random(n_stars)
    
    mags = fake_locus_mags(pos)
    mags += np.random.normal(size=mags.shape, scale=sigma)
    
    n_bad = int(bad_frac*n_stars)
    mags[:n_bad, :] += np.random.normal(size=(n_bad, mags.shape[1]), scale=bad_scale*sigma)
    
    colors = -np.diff(mags, axis=1)
    
    return colors


def test_fit_locus():
    x = gen_fake_locus(n_stars=10000, sigma=0.01, bad_frac=0.1, bad_scale=20.)
    
    # Fit stellar locus
    mag_anchors = fake_locus_mags(np.array([0., 1.]))
    r = -np.diff(mag_anchors, axis=1)
    
    d_max = [0.25 for i in xrange(x.shape[1])]
    
    r = fit_locus(x, r, d_max, max_iter=100, n_sigma=3.0, clip=1.)
    
    r_ref = refine_locus(x, r, d_max, clip=1., n_iter=5)
    
    # Plot fit
    mag_anchors = fake_locus_mags(np.linspace(0., 1., 1000))
    color_anchors = -np.diff(mag_anchors, axis=1)
    n_c = color_anchors.shape[1]
    
    from mpl_toolkits.axes_grid1 import Grid
    
    fig = plt.figure()
    grid = Grid(fig, 111, nrows_ncols=(n_c-1,n_c-1),
                axes_pad=0.05, label_mode='L')
    
    for c1 in xrange(n_c-1):
        for c2 in xrange(c1+1, n_c):
            ax = grid[(n_c-1)*(c2-1)+c1]
            
            sc = ax.scatter(x[:,c1], x[:,c2],
                            c='k', edgecolor='none',
                            alpha=0.10, s=10.)
            
            ax.plot(color_anchors[:,c1], color_anchors[:,c2],
                    'b-', lw=2, alpha=0.25)
            
            ax.scatter(r_ref[1:-1,c1], r_ref[1:-1,c2],
                       c='b', s=45., alpha=1.0)
            
            ax.scatter(r[1:-1,c1], r[1:-1,c2],
                       c='g', s=45., alpha=0.75)
    
    plt.show()


def test_locus_scatter():
    n_loci = 3
    
    x = gen_fake_locus()
    
    mag_anchors = fake_locus_mags(np.linspace(0., 1., n_loci+2))
    r = -np.diff(mag_anchors, axis=1)
    r += 0.01 * 2. * (np.random.random(size=r.shape) - 0.5)
    
    k = geom_utils.get_sl_vectors(r[1:-1])
    d_max = [1.0, 1.0, 1.0]
    
    idx = [geom_utils.get_locus_members(x, r, k, p, d_max) for p in xrange(n_loci)]
    
    c = np.zeros(x.shape[0], dtype='f8')
    
    r_p = np.zeros(r.shape, dtype=r.dtype)
    
    for n,i in enumerate(idx):
        idx_tmp, scatter = geom_utils.locus_scatter(x[i], r[n+1], k[n], clip=1.)
        print n, scatter
        
        l = np.where(i)[0][idx_tmp]
        
        r_p[n+1] = np.mean(x[l], axis=0)
        c[l] = n + 1.
    
    dr = r_p - r
    
    #c[idx[2]] = 1.
    
    # Plot color-color diagram
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    sc = ax.scatter(x[:,0], x[:,1],
                    c=c, edgecolor='none',
                    alpha=0.75, s=10.)
    
    ax.scatter(r[:,0], r[:,1],
            c='g', s=45., alpha=1.0)
    
    ax.scatter(r_p[1:-1,0], r_p[1:-1,1],
            c='g', s=45., alpha=1.0)
    
    for n in xrange(k.shape[0]):
        ax.arrow(r[n+1,0], r[n+1,1], 0.25*k[n,0], 0.25*k[n,1],
                 head_width=0.025, head_length=0.05,
                 fc='r', ec='r', alpha=0.75)
        
        ax.arrow(r[n+1,0], r[n+1,1], dr[n+1,0], dr[n+1,1],
                 head_width=0.01, head_length=0.02,
                 fc='g', ec='g', alpha=0.50)
    
    fig.colorbar(sc)
    
    # Plot in locus coordinates
    fig = plt.figure()
    
    for n,i in enumerate(idx):
        u, x_p = geom_utils.locus_transform(x[i], r[n+1], k[n])
        l, scatter = geom_utils.locus_scatter(x[i], r[n+1], k[n], clip=1.)
        mu = np.mean(x_p, axis=0)
        
        c = np.zeros(x_p.shape[0], dtype='f8')
        c[l] = 1.
        
        ax = fig.add_subplot(1, n_loci, n+1, aspect='equal')
        ax.scatter(x_p[:, 0], x_p[:, 1], c=c, edgecolor='none')
        ax.axhline(mu[1]+scatter, color='r', lw=2., alpha=0.75)
        ax.axhline(mu[1]-scatter, color='r', lw=2., alpha=0.75)
    
    plt.show()
    
    
    plt.show()


def test_get_locus_members():
    n_loci = 15
    
    x = gen_fake_locus()
    
    mag_anchors = fake_locus_mags(np.linspace(0., 1., n_loci+2))
    r = -np.diff(mag_anchors, axis=1)
    r += 0.01 * 2. * (np.random.random(size=r.shape) - 0.5)
    
    k = geom_utils.get_sl_vectors(r[1:-1])
    d_max = [0.20, 0.20, 0.20]
    
    idx = [geom_utils.get_locus_members(x, r, k, p, d_max) for p in xrange(n_loci)]
    
    c = np.zeros(x.shape[0], dtype='f8')
    
    r_p = np.zeros(r.shape, dtype=r.dtype)
    
    for n,i in enumerate(idx):
        r_p[n+1] = np.mean(x[i], axis=0)
        
        c[i] = n + 1.
    
    dr = r_p - r
    
    #c[idx[2]] = 1.
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    sc = ax.scatter(x[:,0], x[:,1],
                    c=c, edgecolor='none',
                    alpha=0.75, s=10.)
    
    ax.scatter(r[:,0], r[:,1],
            c='g', s=45., alpha=1.0)
    
    ax.scatter(r_p[1:-1,0], r_p[1:-1,1],
            c='g', s=45., alpha=1.0)
    
    for n in xrange(k.shape[0]):
        ax.arrow(r[n+1,0], r[n+1,1], 0.25*k[n,0], 0.25*k[n,1],
                 head_width=0.025, head_length=0.05,
                 fc='r', ec='r', alpha=0.75)
        
        ax.arrow(r[n+1,0], r[n+1,1], dr[n+1,0], dr[n+1,1],
                 head_width=0.01, head_length=0.02,
                 fc='g', ec='g', alpha=0.50)
    
    fig.colorbar(sc)
    
    plt.show()


def test_gen_fake_locus():
    colors = gen_fake_locus()
    mag_anchors = fake_locus_mags(np.linspace(0., 1., 1000))
    color_anchors = -np.diff(mag_anchors, axis=1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(colors[:,0], colors[:,1],
               c='k', edgecolor='none',
               alpha=0.5, s=2.)
    
    ax.plot(color_anchors[:,0], color_anchors[:,1],
            'b-', lw=2, alpha=0.5)
    
    plt.show()

def main():
    test_fit_locus()
    
    return 0

if __name__ == '__main__':
    main()

