<p align="center">
  <img src="doc/logo.png" width="60" align="right">
</p>

# OpTaS

OpTaS is an OPtimization-based TAsk Specification library for task and motion planning (TAMP), trajectory optimization, and model predictive control.

# Install

## Via pip
1. `$ python -m pip install 'optas @ git+https://github.com/cmower/optas.git'`

## From source
1. `$ git clone git@github.com:cmower/optas.git`
2. `$ cd optas`
3. `$ pip install --upgrade pip`, ensure `pip` is up-to-date
4. `$ pip install .`

## Build documentation

1. `$ cd /path/to/optas/doc`
2. `$ sudo apt install doxygen`
3. `$ doxygen`
4. Open the documentation in either HTML or PDF:
   - `html/index.html`
   - `latex/refman.pdf`

# Examples
For examples, checkout the [example](example) folder.
