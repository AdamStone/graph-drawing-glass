from __future__ import division
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Euclidean distance functions

def d_vector(i, j, coords, period=None):
    """ Vector from point i to point j. Coords contain coordinates of points
    along each axis as matrix rows, e.g. [[x0, x1, ...], [y0, y1, ...], ...]
    """
    res = np.array([positions[j] - positions[i] for positions in coords])
    """ i.e. [xj-xi, yj-yi, ...]"""

    if period != None:  # periodic boundary conditions
        """ Need to check if shorter distance occurs across boundaries for each
        vector component """

        # for each axis, get the index and distance vector component
        for ax, dist in enumerate(res):

            # get cell width for this axis
            cell_size = period[ax]

            # if dist exceeds half the cell width, it must be shorter across periodic boundary
            if np.abs(dist) > cell_size/2:

                # get specific i and j positions along this axis
                """ e.g. [yi=15, yj=3]"""
                pi, pj = coords[ax][i], coords[ax][j]

                # shift j by cell_size in the proper direction
                res[ax] = pj - np.sign(dist)*cell_size - pi

    return res


def d_sq(i, j, coords, period=None): # TODO periodic bounds
    """ Scalar square Euclidean distance between points i and j. Coords contain
    coordinates of points along each axis, e.g. X=[x0, x1, ...], Y=[y0, y1, ...]
     """
    dv = d_vector(i, j, coords, period)
    return sum(dv**2)


def D_sq(coords, period=None):
    """ Scalar square Euclidean distance matrix between all i, j. Coords contain
    coordinates of points along each axis, e.g. X=[x0, x1, ...], Y=[y0, y1, ...]
    """
    nV = len(coords[0])
    res = np.zeros([nV, nV], dtype=np.uint16)
    for i in range(nV):
        for j in range(nV):
            res[j, i] = res[i, j] = d_sq(i, j, coords, period)
    return res


# Graph distance functions

def D_graph(A):
    D = A.copy()
    L = len(D)
    loops = 0
    for j, row in enumerate(D):

        # starting with nearest-neighbors,
        n = 1

        # until all ij (i!=j) nonzero,

        while np.count_nonzero(row) < L - 1:
            # find indices i of nth-nearest neighbors
            for i, d in enumerate(row):
                if d == n:

                    # get neighbors-of-neighbors
                    r = A[i].copy()

                    # keep i = j 0
                    r[j] = 0

                    # elsewhere multiply by n+1
                    r *= (n+1)

                    # add nth+1-nearest-neighbors to row
                    row += np.where(row==0, r, 0)

            # increment n
            n += 1
            loops += 1
##    print 'D loops ', loops
    return D


def nth_neighbors(j, n, Dg):
    return [i for i, d in enumerate(Dg[j]) if d == n]



# Forces and energy

def f_spring(i, K, Adjacency, coords, species, period=None, C=1, repel=True, attract=True):
    d = lambda j: d_sq(i, j, coords, period)**0.5
    u = lambda j: d_vector(i, j, coords, period) / d(j)

    if attract:
        fs = sum([(d(j) - K[species[i], species[j]])*u(j)
                    for j, edge in enumerate(Adjacency[i])
                    if edge])
    else:
        fs = np.array([0]*len(coords))

    # truncated repulsion between all pairs, 0 for distance >=K and increasing for < K
    if repel:
        def f(j):
            ans = (-C*K[species[i], species[j]]**2 / d_sq(i, j, coords, period) + C)
            if ans <= 0:
                return ans * u(j)
            else:
                return np.zeros([len(coords)])

        fr = sum([f(j)
                  for j in range(len(coords[0]))
                  if j != i])
    else:
        fr = np.zeros([len(coords)])

    return fs + fr

def f_spring_electric(i, K, Adjacency, coords, species, period=None, C=0.2, repel=True, attract=True):

    # repulsion between all pairs
    if repel:
        fr = sum([-C*K[species[i], species[j]]**2 / d_sq(i, j, coords, period) * d_vector(i, j, coords, period)
                    for j in range(len(coords[0]))
                    if j != i])
    else:
        fr = np.zeros([len(coords)])

    # attraction between neighbors
    if attract:
        fa = sum([d_sq(i, j, coords, period)**0.5 / K[species[i], species[j]] * d_vector(i, j, coords, period)
                    for j, edge in enumerate(Adjacency[i])
                    if edge])
    else:
        fa = np.zeros([len(coords)])

    f = fa + fr
    if not np.array_equal(f, 0):
        return f
    else:
        return np.zeros([len(coords)])

def forces(f_func, K, Adjacency, coords, species, period=None, **kwargs):
    result = [f_func(i, K, Adjacency, coords, species, period, **kwargs)
                for i in range(len(coords[0]))]
    return np.array(result)

def energies(forces_array):
    return np.array([sum(f**2) for f in forces_array])

def update_step(step, E, E0, progress=0, t=0.90):
    if E < E0:
        progress += 1
        if progress >= 4:
            progress = 0
            step /= t
    else:
        progress -= 1
        step *= t

    return step, progress



# visualization
def plot(coords, adjacency, nSi, nB, nO, cell=None):
    fig = plt.figure()

    if cell != None:
        aspect = cell[0]/cell[1]
    else:
        aspect = 1

    if len(coords) == 2:
        X, Y = coords
        ax = fig.add_subplot(111, aspect=aspect)
    elif len(coords) == 3:
        X, Y, Z = coords
        ax = fig.add_subplot(111, aspect=aspect, projection='3d')

    # edges
    for j, edge in enumerate(adjacency):
        for i, connection in enumerate(edge[j:]):
            i += j # shift to account for enumerating from [j:]
            if connection:

                if cell is not None: # check if edge crosses boundary
                    pi = np.array([Ax[i] for Ax in coords])
                    pj = np.array([Ax[j] for Ax in coords])
                    dij = d_vector(i, j, coords, cell)
                    pj_ = pi + dij
                    pi_ = pj - dij

                    if np.greater(np.abs(pi - pi_), 1e-10).any() or np.greater(np.abs(pj - pj_), 1e-10).any():
                        # bond crosses a periodic boundary
                        # the direct bond should not be shown
                        # two bonds going off-screen from each point should be shown

                        e1 = [] # edge 1
                        e2 = [] # edge 2

                        # edge endpoints for each coordinate
                        for k in range(len(coords)): # for each dimension
                            e1.append([pi[k], pj_[k]])
                            e2.append([pi_[k], pj[k]])

                        m1 = [pj_[k] - pi[k] for k in range(len(coords))] # slope 2
                        m2 = [pj[k] - pi_[k] for k in range(len(coords))] # slope 1

                        # for each edge, one of the two endpoints should be moved
                        # to the boundary of the cell using the slope

                        # edge 1:
                        factor = 1
                        for k, axis in enumerate(e1):
                            if axis[1] > cell[k]:
                                factor = min((cell[k] - axis[0])/m1[k], factor)
                            if axis[1] < 0:
                                factor = min((0 - axis[0])/m1[k], factor)

##                        print '\nnext edge:'
                        for k, axis in enumerate(e1):
##                            print axis, cell[k]
                            axis[1] = axis[0] + m1[k]*factor
##                            print axis, '\n'

                        # edge 2:
                        factor = 1
                        for k, axis in enumerate(e2):
                            if axis[0] > cell[k]:
                                factor = min((cell[k] - axis[1])/m1[k], factor)
                            if axis[0] < 0:
                                factor = min((0 - axis[1])/m1[k], factor)

##                        print '\nnext edge:'
                        for k, axis in enumerate(e2):
##                            print axis, cell[k]
                            axis[0] = axis[1] + m1[k]*factor
##                            print axis, '\n'

                        """ can probably condense into one function, clip() """

                        ax.plot(*e1, color='gray', lw=0.5)
                        ax.plot(*e2, color='gray', lw=0.5)
                    else:
                        # just show the direct bond within the boundaries
                        ax.plot(*[[Ax[i], Ax[j]] for Ax in coords], color='gray', lw=0.5)
                else:
                    # just show the direct bond within the boundaries
                    ax.plot(*[[Ax[i], Ax[j]] for Ax in coords], color='gray', lw=0.5)

    # atoms
    ##ax.plot(X[:nSi], Y[:nSi], 'go', ms=8)               # Si
    ax.plot(*[Ax[:nSi] for Ax in coords], color='g', marker='o', ms=8, lw=0)               # Si
    ##ax.plot(X[nSi:nSi+nB], Y[nSi:nSi+nB], 'bo', ms=5)   # B
    ax.plot(*[Ax[nSi:nSi+nB] for Ax in coords], color='b', marker='o', ms=5, lw=0)   # B
    ##ax.plot(X[nSi+nB:nSi+nB+nO], Y[nSi+nB:nSi+nB+nO],   # O
    ##        marker='o', lw=0, color='white', ms=15)
    ax.plot(*[Ax[nSi+nB:nSi+nB+nO] for Ax in coords],   # O
            marker='o', color='white', ms=15, lw=0)

    # labels
    ##for i, (x, y) in enumerate(zip(X, Y)):
    ##    try:
    ##        ax.text(x, y+cell[1]/40, i, color='black')
    ##    except:
    ##        ax.text(x, y+1, i, color='black')

    if cell != None:
        plt.xlim(0, cell[0])
        plt.ylim(0, cell[1])
        try:
            ax.set_zlim(0, cell[2])
        except:
            pass
    plt.show()



