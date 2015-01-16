from __future__ import division
import numpy as np
import random

from graph import D_graph

class Borosilicate(object):
    def __init__(self, SiO2, B2O3):
        # composition
        nSi = self.nSi = SiO2
        nB = self.nB = int(B2O3/2)
        nO = self.nO = nSi*2 + int(nB*3/2)
        nV = self.nV = nSi + nB + nO

        # matrix coordinates of each atom
        Si = self.Si = np.array(range(nSi))
        B = self.B = np.array(range(nB)) + nSi
        O = self.O = np.array(range(nO)) + nSi + nB
        V = self.V = np.hstack([Si, B, O])

        # species index array
        self.species = np.array([0]*nSi + [1]*nB + [2]*nO)

        # build adjacency matrix with random topology
        qSi = list(Si) * self.z(Si)
        qB = list(B) * self.z(B)
        qO = list(O) * self.z(O)
        [random.shuffle(q) for q in [qO]]

        E = []
        stalled = 0
        qSi_last = []
        E0 = E[:]
        qO0 = qO[:]
        while qSi:
            qSi_last = qSi[:]
            if stalled > 5:
                E = E0[:]
                qSi = list(Si) * self.z(Si)
                qO = qO0[:]
                [random.shuffle(q) for q in [qO]]
                stalled = 0
                qSi_last = []
            si = qSi.pop()
            o = qO.pop()
            # avoid duplicates
            if [si, o] in E:
                qO = [o] + qO
                qSi.append(si)
            else:
                # avoid edge-sharing
                si_neighbors = []
                o_neighbors = []
                for edge in E:
                    if edge[0] == si:
                        si_neighbors.append(edge[1])
                    if edge[1] == o:
                        o_neighbors.append(edge[0])
                si_second_neighbors = []
                for edge in E:
                    if edge[1] in si_neighbors:
                        si_second_neighbors.append(edge[0])
                valid = True
                for o_nb in o_neighbors:
                    if o_nb in si_second_neighbors:
                        valid = False
                        qO = [o] + qO
                        qSi.append(si)
                        break
                if valid:
                    E.append([si, o])
            if qSi == qSi_last:
                stalled += 1

        stalled = 0
        qB_last = []
        E0 = E[:]
        qO0 = qO[:]
        while qB:
            qB_last = qB[:]
            if stalled > 5:
                E = E0[:]
                qB = list(B) * self.z(B)
                qO = qO0[:]
                [random.shuffle(q) for q in [qO]]
                stalled = 0
                qB_last = []
            b = qB.pop()
            o = qO.pop()
            # avoid duplicates
            if [b, o] in E:
                qO = [o] + qO
                qB.append(b)
            else:
                # avoid edge-sharing
                b_neighbors = []
                o_neighbors = []
                for edge in E:
                    if edge[0] == b:
                        b_neighbors.append(edge[1])
                    if edge[1] == o:
                        o_neighbors.append(edge[0])
                b_second_neighbors = []
                for edge in E:
                    if edge[1] in b_neighbors:
                        b_second_neighbors.append(edge[0])
                valid = True
                for o_nb in o_neighbors:
                    if o_nb in b_second_neighbors:
                        valid = False
                        qO = [o] + qO
                        qB.append(b)
                        break
                if valid:
                    E.append([b, o])
            if qB == qB_last:
                stalled += 1
        self.E = E

    # coordination number
    def z(self, i):
        try:
            i = i[0]
        except:
            pass

        if i in self.Si:
            return 4
        elif i in self.B:
            return 3
        else:
            return 2

    def get_adjacencyMatrix(self):
        try:
            return self.adjacencyMatrix
        except:
            nV = len(self.V)
            A = np.zeros([nV, nV], dtype=np.uint8) # can be np.bool_, but uint prints 0s and 1s
            for i, j in self.E:
                A[i, j] = A[j, i] = 1
            self.adjacencyMatrix = A
            return A

    def get_graphDistanceMatrix(self):
        try:
            return self.graphDistanceMatrix
        except:
            try:
                A = self.adjacencyMatrix
            except:
                A = self.get_adjacencyMatrix()
            Dg = D_graph(A)
            self.graphDistanceMatrix = Dg
            return Dg

    def get_graphDiameter(self):
        try:
            Dg = self.graphDistanceMatrix
        except:
            Dg = self.get_graphDistanceMatrix()

        diam = np.max(Dg)
        self.graphDiameter = diam
        return diam


    def get_angularAdjacency(self):
        try:
            return self.angularAdjacency
        except:
            pE_Si = []
            try:
                A = self.adjacencyMatrix
            except:
                A = self.get_AdjacencyMatrix()
            for i, si in enumerate(self.Si):

                neighbs = [j for j, edge in enumerate(A[i]) if edge]

                # add pseudo-edge between all 3 pairwise neighbors
                o1, o2, o3, o4 = [neighbs[k] for k in range(self.z(self.Si))]
                pE_Si.append([o1, o2])
                pE_Si.append([o1,o3])
                pE_Si.append([o1,o4])
                pE_Si.append([o2,o3])
                pE_Si.append([o2,o4])
                pE_Si.append([o3,o4])


            pE_B = []
            for i, b in enumerate(self.B):
                i += len(self.Si) # boron occurs after Si in the adjacency matrix

                neighbs = [j for j, edge in enumerate(A[i]) if edge]

                # add pseudo-edge between all 3 pairwise neighbors
                o1, o2, o3 = [neighbs[k] for k in range(self.z(self.B))]
                pE_B.append([o1, o2])
                pE_B.append([o1,o3])
                pE_B.append([o2, o3])


            # pseudo-edge adjacency
            nV = len(self.V)
            pA_B = np.zeros([nV, nV], dtype=np.uint8) # dtype can be np.bool_, but uint prints 0s and 1s
            for i, j in pE_B:
                pA_B[i, j] = pA_B[j, i] = 1

            pA_Si = np.zeros([nV, nV], dtype=np.uint8) # dtype can be np.bool_, but uint prints 0s and 1s
            for i, j in pE_Si:
                pA_Si[i, j] = pA_Si[j, i] = 1

            aa = {'SiO2': pA_Si, 'B2O3': pA_B}
            self.angularAdjacency = aa
            return aa