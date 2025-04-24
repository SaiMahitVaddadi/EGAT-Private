from itertools import chain, groupby

import numpy as np

class BFSTree(object):
    __slots__ = ("tree", "visited", "bonds", "atoms")

    def __init__(self, mol):
        self.tree = {}
        self.visited = set()

        self.bonds = {}
        for b in mol.GetBonds():
            s = b.GetBeginAtomIdx()
            d = b.GetEndAtomIdx()
            t = b.GetBondType()

            self.bonds[s, d] = t
            self.bonds[d, s] = t

        self.atoms = [
            (a.GetAtomicNum(), a.GetDegree(), a.GetNeighbors()) for a in mol.GetAtoms()
        ]

    def reset(self, i):
        self.tree.clear()
        self.visited.clear()

        self.tree[i] = ()
        self.visited.add(i)

    def expand(self):
        self._expand(self.tree)

    def _expand(self, tree):
        for src, dst in list(tree.items()):
            self.visited.add(src)

            if dst is ():
                tree[src] = {
                    n.GetIdx(): ()
                    for n in self.atoms[src][2]
                    if n.GetIdx() not in self.visited
                }

            else:
                self._expand(dst)

    def _code(self, tree, before, trail):
        if len(tree) == 0:
            yield trail

        else:
            for src, dst in tree.items():

                code = []
                if before is not None:
                    bt = self.bonds[before, src]
                    code.append(bt)

                code.append(self.atoms[src][:2])

                nxt = tuple(chain(trail, code))
                for t in self._code(dst, src, nxt):
                    yield t

    def get_code(self, i, order):
        self.reset(i)

        for _ in range(order):
            self.expand()
