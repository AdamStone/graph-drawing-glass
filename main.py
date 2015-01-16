from __future__ import division
import random
import numpy as np

from graph import Borosilicate, forces, energies, f_spring, f_spring_electric, update_step, plot, nth_neighbors


""" General procedure:

1. Generate an initial graph G0, with vertices V corresponding to atoms of a
given glass composition and edges E corresponding to nearest-neighbor bonds, and
subject to a set of specified topological constraints

     The constraints are basic bonding rules (e.g. Si bonds to 4 O, O bonds to
     two Si, ...) but may also include specific characteristics of interest such
     as phase separation, "avoidance rules," particular distributions of ring
     sizes, etc.

2. Partition vertices into nested sets V0 > V1 > ... > Vk with varying
coarseness, and |Vk| = 3

For each Vi starting at Vk,

3. Assign initial positions based on graph distance and positions of already-placed nodes.

4. Apply an energy minimization iteration to refine the positions of placed nodes.

5. Check whether result is still compatible with a physical glass network with
fixed bond lengths; if so, move to next Vi.

6. If not, apply a multiple edge switching and repeat the current Vi until a
suitable structure is obtained.

"""


G = Borosilicate(6, 20)


# partition vertices into maximally independent vertex sets
MIVS = [set(G.V), set(G.B)]
current = set([])

Dg = G.get_graphDistanceMatrix()
diam = G.get_graphDiameter()

retries = 0
while True: # outer loop over all MIVS
##    print 'MIVS:\n', MIVS
    """ partition each complete subset into two smaller independent subsets """

    n = len(MIVS) # last complete subset contains nth nearest neighbors
    last = MIVS[-1].copy()
    if len(last) < 4 or len(last) <= retries:
        break

    # pop a starting node in last complete subset
    print last, list(last), retries
    start = list(last)[retries]
    last.remove(start)
    current.add(start)

    reset = False
    while last: # inner loop over current MIV
        print '\nlast:\n', last
        print 'current:\n', current

        to_add = set([])

        # find nth nearest neighbors of each node in current
        for node in current:
            n_nb = nth_neighbors(node, 2**n, Dg)
            print 'node, n_nb:', node, n_nb

            # keep any that have d_graph >= n for all others in current and n_nb
            while n_nb:
                nb = n_nb.pop()
                if nb in MIVS[-1]:
                    dg = [Dg[nb, cr] for cr in current]
                    dg += [Dg[nb,nb2] for nb2 in to_add]
                    dg = np.array(dg)
                    print 'graph distance to current', dg
                    if len(np.where(dg < 2**n)[0]) == 0:
                        print 'attempt to add ', nb
                        to_add.add(nb)
                    else:
                        print 'rejecting {} for d_graph < n'.format(nb)
                    if nb in last:
                        last.remove(nb)
                    print n_nb
                else:
                    print 'nb not in superset'

            print 'checking to_add'
            if not to_add:
                if len(current) == 1:
                    print 'none to add from initial node; retrying'
                    reset = True
                    last = False
                    current = set([])
                    retries += 1
                    break
                else:
                    print 'maximum reached for this branch'
                    last = False
                    on = 0
                    break
            else:
                print 'to_add True:', to_add

        current.update(to_add)

    # break the outer loop when the smallest MIV is reached or reset required
    if reset:
        print 'retrying with new starting point'
        pass
    else:
        MIVS.append(current)
        current = set([])

for VS in MIVS:
    print VS


# 2D or 3D?
dimensions = 3

# periodic boundary conditions?
cell = [diam/dimensions] * dimensions
##cell = None

# assign random positions
X0 = [random.random()*diam for v in G.V]
Y0 = [random.random()*diam for v in G.V]
Z0 = [random.random()*diam for v in G.V]
coords = np.vstack([X0, Y0, Z0][:dimensions])


K = 1
pK = K*1.7320508075688772 # sqrt(3); triangle r/2*sqrt(3) = a/2

K_Si = 1.2
pK_Si = K_Si*1.6329931618554523 # sqrt(8/3); tetrahedra r = sqrt(8/3)*a


Si_O = O_Si = 1.2
B_O = O_B = 1

Si_Si = np.average([Si_O, 2*Si_O])
B_B = np.average([B_O, 2*B_O])
O_O = Si_B = B_Si = np.average([Si_Si, B_B])
""" although O-O, Si-Si, or B-B direct bonds don't occur, the K value also gives the
repulsion distance between all pairwise atoms in addition to the spring length.
These should always be second-nearest-neighbors, so the repulsion should be
larger than the nearest-neighbor lengths, but less than 2x them since O angles
< 180degrees are permitted. For a rough starting point we take the average
between 1st neighbor and 2x first neighbor distances, but a full consideration
of bond angles would be more proper. """


K = np.array([[Si_Si, B_Si, O_Si],
              [Si_B,  B_B,  O_B],
              [Si_O,  B_O,  O_O]])

pK_B = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, B_O*1.7320508075688772]])

pK_Si = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, Si_O*1.6329931618554523]])




A = G.get_adjacencyMatrix()



# start with coarsest set and assign positions roughly based on graph distance
positioned = set([])
X = np.zeros([len(G.V)])
Y = np.zeros([len(G.V)])
##X = np.array([len(G.V)])
while len(MIVS) > 0:
    # get coarsest vertex set remaining and cast to list
    vs = list(MIVS.pop())

    # minimum graph distance separating nodes in this set
    gd = 2**(len(MIVS))

    # build 'adjacency matrix' of nodes with minimum graph distance
    A_vs = np.zeros((len(vs), len(vs)), dtype=np.uint8)
    for i, inode in enumerate(vs):
        for j, jnode in enumerate(vs):
            if j > i and Dg[i, j] == gd: # symmetric matrix, only consider one half
                A_vs[i, j] = A_vs[j, i] = 1

    if len(positioned) == 0:
        # random starting point:
        node = vs.pop()
        X[node] = random.random()*diam/2
        Y[node] = random.random()*diam/2
        positioned.add(node)
    else:
        # position relative to already-positioned neighbors
        pass







""" modified spring-electric model """
step = 1
progress = 0
E = 1e10
count = 0

# strength of repulsion relative to spring force
C = .001

""" C ~ 1 gives well-separated nodes at the expense of bond angles and ideal distances,
i.e. large distortion. C ~ 0.001 retains bond lengths and shapes of unit polyhedra
by allowing atoms to occur close together. To obtain well-separated polyhedra near
their ideal shape, the topology must be optimized. """

print 'Target energy:', Si_O*.001*len(G.V)

species = G.species
aa = G.get_angularAdjacency()
pA_B = aa['B2O3']
pA_Si = aa['SiO2']

while True:
    count += 1

    F = [forces(f_spring, K, A, coords, species, cell, C=C)   # edge forces
     + forces(f_spring, pK_B, pA_B, coords, species, cell, C=C, repel=False) # B pseudoedge forces
     + forces(f_spring, pK_Si, pA_Si, coords, species, cell, C=C, repel=False) # Si pseudoedge forces
    ][0] # (wrapped in list only to facilitate commenting out pseudoedge lines)

    E_arr = energies(F) # energy per node (F**2)
    E, E0 = np.sum(E_arr), E # total energy
    print E, count, progress
    if E < Si_O*.001*len(G.V) or count > 200 or progress < -5:
        break

    # broadcast E_arr (i.e. F_sq) to same shape as F
    Fsq_vector = np.vstack([E_arr]*len(coords)).transpose()

    # shift coords by step * unit vector
    """ is there a sensible way to include force magnitude somehow?"""
    shift = step*F/Fsq_vector**(0.5)
    coords += shift.transpose()

    # if cell boundaries were defined, enforce periodicity
    if cell != None:
        for i, length in enumerate(cell):
            coords[i] %= length

    step, progress = update_step(step, E, E0, progress)

plot(coords, A, G.nSi, G.nB, G.nO, cell)
