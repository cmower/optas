import collections
import casadi as cs
import numpy as np
from typing import Union, List

SX = cs.casadi.SX
DM = cs.casadi.DM

class SXContainer(collections.OrderedDict):

    """Container for SX variables"""

    is_discrete = {} # Dict[str, bool]: labels for each item in dict, true means variables are discrete

    def __add__(self, other):
        """Add two SXContainer's"""
        assert isinstance(other, SXContainer), f"cannot add SXContainer with a variable of type {type(other)}"
        out = SXContainer()
        for label, value in self.items():
            out[label] = value
        for label, value in other.items():
            out[label] = value
        out.is_discrete = {**self.is_discrete, **other.is_discrete}
        return out

    def __setitem__(self, label: str, value: SX) -> None:
        """Set new SX item"""
        assert isinstance(value, (SX, float)), f"value must be of type casadi.casadi.SX/float, not {type(value)}"
        if label in self: raise KeyError(f"'{label}' already exists")
        super().__setitem__(label, cs.SX(value))
        self.is_discrete[label] = False  # assume non-discrete, otherwise variable_is_discrete(..) should be called

    def variable_is_discrete(self, label: str) -> None:
        assert label in self, f"'{label}' was not found"
        self.is_discrete[label] = True

    def has_discrete_variables(self) -> bool:
        """True if any of the variables in container are discrete, False otherwise."""
        return any(self.is_discrete.values())

    def discrete(self) -> List[bool]:
        """Returns a list containing discrete classification of each variable."""
        out = []
        for label, value in self.items():
            m, n = values.shape
            out += [self.is_discrete[label]]*(m*n)
        return out

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
            out_vec = cs.vertcat(out_vec, cs.vec(v))
        return out_vec

    def zero(self) -> dict:
        """Return dictionary containing layout of container with zero values"""
        return {label: cs.DM.zeros(*value.shape) for label, value in self.items()}
