# pyinvk

![Alt Text](https://raw.githubusercontent.com/cmower/pyinvk/master/fig8.gif)

`pyinvk` is a library that allows you to setup an inverse kinematic problem with optional constraints.
The package interfaces with several open-source and commerical solvers.

`pyinvk` builds on top of [CasADi](https://web.casadi.org/).
This allows you to compute deriviates of the FK to arbitrary order.

`pyinvk` interfaces with CasADi solvers via `nlpsol` (e.g. IPOPT, SNOPT), and also Scipy optimization routines via the [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) method (see *method* parameter for full list of available solvers).
New interfaces can be added by implementing a class that inherits from `Solver` in [`solvers.py`](https://github.com/cmower/pyinvk/blob/master/pyinvk/solver.py).

See the [examples](https://github.com/cmower/pyinvk/tree/master/example).

# Install

## Pip

```
pip install pyinvk
```

## From source

1. `$ git clone git@github.com:cmower/pyinvk.git`
2. `$ cd pyinvk`
3. `$ pip install .`
