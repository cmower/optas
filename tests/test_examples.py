import os
import sys
import pathlib
import importlib

optas_path = pathlib.Path(
    __file__
).parent.parent.resolve()  # path to current working directory
examples_path = optas_path / "example"

sys.path.append(str(examples_path.absolute()))


def import_module(path):
    spec = importlib.util.spec_from_file_location("user_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return module


def load_main_function(filename):
    path = examples_path / filename

    # Get main function handle from filename
    module = import_module(path)
    main = getattr(module, "main")

    return main


def test_dual_arm():
    main = load_main_function("dual_arm.py")
    assert main(gui=False) == 0


def test_planar_idk():
    main = load_main_function("planar_idk.py")
    assert main() == 0


def test_planar_ik():
    main = load_main_function("planar_ik.py")
    assert main() == 0


def test_figure_eight_plan():
    main = load_main_function("figure_eight_plan.py")
    assert main(gui=False) == 0


def test_figure_eight_plan_6dof():
    main = load_main_function("figure_eight_plan_6dof.py")
    assert main(gui=False) == 0


def test_point_mass_mpc():
    main = load_main_function("point_mass_mpc.py")
    assert main(show=False) == 0


def test_point_mass_planner():
    main = load_main_function("point_mass_planner.py")
    assert main(show=False) == 0


def test_pushing():
    main = load_main_function("pushing.py")
    assert main(gui=False) == 0


def test_pushing():
    main = load_main_function("pushing.py")
    assert main(gui=False) == 0


def test_pybullet_api():
    main = load_main_function("pybullet_api.py")
    assert main(gui=False) == 0


def test_simple_joint_space_planner():
    main = load_main_function("simple_joint_space_planner.py")
    assert main(gui=False) == 0


# def test_example():
#     path = examples_path / "example.py"
#     module = import_module(path)
