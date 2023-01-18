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

# Examples
For examples, checkout the [example](example) folder.

# Citation

If you use OpTaS in your work, please consider including the following citation.

```
@inproceedings{Mower2023,
  author={Mower, Christopher E. and Moura, João and Zamani Behabadi, Nazanin and Vijayakumar, Sethu and Vercauteren, Tom and Bergeles, Christos},
  booktitle={2023 International Conference on Robotics and Automation (ICRA)},
  title={OpTaS: An Optimization-based Task Specification Library for Trajectory Optimization and Model Predictive Control},
  year={2023},
  url = {https://github.com/cmower/optas},
}
```

# Acknowledgement

This research received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement No. 101016985 (FAROS).
Further, this work was supported by core funding from the Wellcome/EPSRC [WT203148/Z/16/Z; NS/A000049/1].
Tom Vercauteren is supported by a Medtronic / RAEng Research Chair [RCSRF1819\7\34], and Christos Bergeles by an ERC Starting Grant [714562].
This research is supported by Kawada Robotics Corporation.
