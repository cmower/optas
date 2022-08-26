<p align="center">
  <img src="doc/logo.png" width="60" align="right">
</p>

# OPTAS

`OPTAS` is a library that allows you to setup an inverse kinematic problem for arbitrary time-horizon and optional constraints.
The package interfaces with several open-source and commerical optimization solvers and is built on top of [CasADi](https://web.casadi.org/).
This allows you to compute deriviates of any forward function to arbitrary order (including the foward kinematics).
Additionally, any number of robots can be included in the optimization problem by supplying their URDF.

`OPTAS` interfaces with a number of solvers for solving QP and NLP problems with/without constraints.
The optimization builder class selects the appropriate optimization problem class for your given cost function and constraints.
Depending on the problem class you can interface with a number of solvers:
- Solvers that interface with CasADi `qpsol`/`nlpsol` (e.g. IPOPT, SNOPT, qpOASES, KINTRO)
- Scipy routines via the [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) method.
- [OSQP solver](https://osqp.org/)
- [CVXOPT](https://cvxopt.org/index.html)

New interfaces can be added by implementing a class that inherits from `Solver` in [`solvers.py`](https://github.com/cmower/optas/blob/master/optas/solver.py).

See the [examples](https://github.com/cmower/optas/tree/master/example).

# Install

## Pip

```
pip install optaslib
```

## From source

1. `$ git clone git@github.com:cmower/optas.git`
2. `$ cd optas`
3. `$ pip install .`

# Build documentation

1. `cd doc/`
2. `doxygen`
