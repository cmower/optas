# Examples

To run the examples out of the box, you must have the [ROS-Pybullet Interface](https://github.com/cmower/ros_pybullet_interface) installed.
Start the simulator using 
```
  $ roslaunch rpbi_examples run_kuka.launch
```

In another terminal, you can now run the examples.

## `example_kuka_lwr.py`

```
  $ python example_kuka_lwr.py SIDE INTERFACE SLSQP N
```
- `SIDE`, `1` (left) or `-1` (right)
- `INTERFACE`, `casadi` or `scipy`
- `METHOD`
  - when `INTERFACE=casadi`, `ipopt`, `snopt`, `knitro` or any other plugin solver name
  - when `INTERFACE=scipy`, `method` as in [`scipy.optimize.minimize` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- `N`, discretization, positive (non-zero) integer


## Comparisons

# Best comparisons

![Alt Text](fig/time.png)

![Alt Text](fig/err.png)

![Alt Text](fig/err_eul.png)

![Alt Text](fig/jdiff.png)
