#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  geom_utils.py
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


#
# Gram-Schmidt Orthogonalization
#

def proj_onto_multiple(u, v):
    '''
    Project a single vector, v, onto a set of vectors, u.
    '''
    
    uv = np.einsum('ij,j->i', u, v)
    uu = np.sum(u**2, axis=1)
    
    return np.einsum('i,ij->ij', uv/uu, u)


def proj_from_multiple(u, v):
    '''
    Project multiple vectors, v, onto a single vector, u.
    '''
    
    uv = np.einsum('k,ik->i', u, v)
    uu = np.sum(u**2)
    
    return np.einsum('i,j->ij', uv, u/uu)


def proj_single(u, v):
    '''
    Project a single vector, v, onto a single vector, u.
    '''
    
    return np.sum(u*v) / np.sum(u**2) * u


def gram_schmidt(v):
    '''
    Return a set of orthonormal vectors that span the same space as
    the set of vectors v.
    '''
    
    u = np.empty(v.shape, dtype=v.dtype)
    u[:] = v[:]
    
    for i in xrange(u.shape[0]-1):
        u[i] /= np.sqrt(np.sum(u[i]**2))
        u[i+1:] -= proj_from_multiple(u[i], u[i+1:])
    
    u[-1] /= np.sqrt(np.sum(u[-1]**2))
    
    return u


#
# Stellar locus coordinates
#

def get_sl_vectors(r):
    '''
    Return the locus vector, k_p, at each locus, r_p.
    '''
    
    k = np.empty(r.shape, dtype=r.dtype)
    
    k[1:-1] = r[2:] - r[:-2]
    k[0] = r[1] - r[0]
    k[-1] = r[-1] - r[-2]
    
    return k


def locus_transform(x, r_p, k_p):
    '''
    Transform a set of stellar colors, x, to the coordinate system of
    the locus r_p, defined by the local locus vector, k_p.
    
    Inputs:
      x    Stellar colors. Shape = (# of stars, # of colors).
      r_p  Coordinates of locus point. Shape = (# of colors).
      k_p  Locus vector, tangent to stellar locus. Shape = (# of colors)
    
    Returns:
      u    Basis vectors. Shape = (vector #, color).
      x_p  Transformed colors. Shape = (# of stars, # of colors).
    '''
    
    # Generate local basis vectors
    n_dim = k_p.size
    u = np.random.random((n_dim, n_dim))
    u[0, :] = k_p[:]
    u = gram_schmidt(u)
    
    # Center stars on locus
    x_p = np.empty(x.shape, dtype=x.dtype)
    x_p[:] = x[:]
    
    for i in xrange(x_p.shape[1]):
        x_p[:, i] -= r_p[i]
    
    # Project stars onto local basis
    x_p = np.einsum('nj,ij->ni', x_p, u)
    
    return u, x_p


def get_locus_members(x, r, k, p, d_max):
    '''
    Determine which stars in x belong to locus p. The set of vectors
    r contains the starting vector, the locus positions, and the ending
    vector, in that order. The set of vectors k contains the locus
    vectors.
    '''
    
    # Center coordinates on locus
    x_p = np.empty(x.shape, dtype=x.dtype)
    x_p[:] = x[:]
    
    for i in xrange(x_p.shape[1]):
        x_p[:, i] -= r[p+1, i]
    
    # Determine distance from locus center, projected onto locus vector
    d_k = np.einsum('nj,j->n', x_p, k[p])
    
    # Determine bounds on distance along locus vector
    d_lower = np.sum(0.5 * (r[p] - r[p+1]) * k[p])
    d_upper = np.sum(0.5 * (r[p+2] - r[p+1]) * k[p])
    
    # Select stars between the bounding orthogonal surfaces
    idx = (d_k > d_lower) & (d_k < d_upper)
    
    # Filter out stars that are too far away from locus center
    for i in xrange(x.shape[1]):
        idx = idx & (np.abs(x_p[:, i]) < d_max[i])
    
    return idx


def locus_scatter(x, r_p, k_p, clip=2.):
    # Coordinates transverse to locus vector
    u, x_p = locus_transform(x, r_p, k_p)
    x_p = x_p[:, 1:]
    
    # Scatter in the transverse plane
    mu = np.mean(x_p, axis=0)
    cov = np.cov(x_p, rowvar=0)
    
    inv_cov = None
    
    if len(cov.shape) == 0:
        inv_cov = np.array([[1./(cov + 1.e-3)]])
    else:
        inv_cov = np.linalg.inv(cov + 1.e-3*np.identity(cov.shape[0]))
    
    # Clip stars beyond given number of ellipse radii
    for i in xrange(x_p.shape[1]):
        x_p[:, i] -= mu[i]
    
    tmp = np.einsum('ni,ij->nj', x_p, inv_cov)
    d = np.einsum('nj,nj->n', tmp, x_p)
    
    idx = (d < clip**2.)
    
    # Recompute scatter for clipped sample
    cov = np.cov(x_p[idx], rowvar=0)
    
    if len(cov.shape) == 0:
        return idx, np.sqrt(np.abs(cov))
    
    eival = np.linalg.eigvals(cov)
    
    return idx, np.sqrt(np.max(np.abs(eival)))


#
# Test functions
#

def test_gram_schmidt(n_dim=2):
    import matplotlib.pyplot as plt
    
    v = 2. * (np.random.random((n_dim, n_dim)) - 0.5)
    u = gram_schmidt(v)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    
    for uu,vv in zip(u,v):
        ax.arrow(0, 0, vv[0], vv[1],
                 head_width=0.05, head_length=0.1,
                 fc='k', ec='k', alpha=0.5)
        ax.arrow(0, 0, uu[0], uu[1],
                 head_width=0.05, head_length=0.1,
                 fc='k', ec='k', alpha=1.0)
    
    p = proj_single(v[0], v[1])
    ax.arrow(0, 0, p[0], p[1],
             head_width=0.05, head_length=0.1,
             fc='r', ec='r', alpha=1.0)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    plt.show()


def main():
    test_gram_schmidt(n_dim=2)
    
    return 0

if __name__ == '__main__':
    main()

