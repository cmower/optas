import collections
import casadi as cs
import numpy as np
from typing import Union

SX = cs.casadi.SX
DM = cs.casadi.DM

class SXContainer(collections.OrderedDict):

    def __add__(self, other):
        """Add two SXContainer's"""
        assert isinstance(other, SXContainer), f"cannot add SXContainer with a variable of type {type(other)}"
        out = SXContainer()
        for label, value in self.items():
            out[label] = value
        for label, value in other.items():
            out[label] = value
        return out

    def __setitem__(self, label: str, value: SX) -> None:
        """Set new SX item"""
        assert isinstance(value, SX), "value must be of type casadi.casadi.SX"
        if label in self: raise KeyError(f"'{label}' already exists")
        super().__setitem__(label, value)

    def vec(self) -> SX:
        """Vectorize SXContainer"""
        values = list(cs.vec(value) for value in self.values())
        return cs.vertcat(*values)

    def numel(self) -> int:
        """Return the number of elements"""
        return self.vec().numel()

    def vec2dict(self, vec: Union[DM, np.ndarray, list, tuple]) -> dict:
        """Convert vector with same layout as SXContainer to dict"""
        vec = cs.DM(vec)  # vec could be np.ndarray or cs.casadi.DM or list/tuple
        out_dict = {}
        for label, value in self.items():
            m, n = value.shape
            mn = m*n
            out_dict[label] = cs.reshape(vec[:mn], m, n)
            vec = vec[mn:]
        return out_dict

    def dict2vec(self, d):
        """Vectorize dictionary with same layout as container"""
        out_vec = cs.DM()
        for label, value in self.items():
            v = d.get(label, cs.DM.zeros(*value.shape))
            out_vec = cs.vertcat(out_vec, v)
        return out_vec

    def zero(self) -> dict:
        """Return dictionary containing layout of container with zero values"""
        return {label: cs.DM.zeros(*value.shape) for label, value in self.items()}
