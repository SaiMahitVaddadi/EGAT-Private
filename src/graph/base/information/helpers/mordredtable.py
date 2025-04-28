from __future__ import division

import os

import six
import numpy as np

class PeriodicTable(object):
    __slots__ = ("data",)

    _datadir = os.path.join(os.path.dirname(__file__), "data")

    def __init__(self, data=None):
        self.data = data

    @classmethod
    def load(cls, name, conv=float):
        def read(v):
            if "-" in v:
                return np.nan

            try:
                return conv(v)
            except ValueError:
                return

        self = cls()

        with open(os.path.join(cls._datadir, name)) as file:
            self.data = [
                v for v in (read(l.split("#")[0]) for l in file) if v is not None
            ]

        return self

    def __getitem__(self, i):
        if i < 1:
            return np.nan

        try:
            return self.data[i - 1]
        except IndexError:
            return np.nan

    def map(self, f):
        new = self.__class__()
        new.data = [f(d) for d in self.data]
        return new