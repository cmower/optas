from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyoptas",
    description="An optimization-based task specification library for task and motion planning (TAMP), trajectory optimization, and model predictive control.",
    version="1.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmower/optas",
    project_urls={
        "Bug Tracker": "https://github.com/cmower/optas/issues",
    },
    author="Christopher E. Mower",
    author_email="christopher.mower@kcl.ac.uk",
    license="Apache License, Version 2.0",
    packages=["optas"],
    install_requires=[
        "numpy",
        "scipy",
        "casadi",
        "urdf-parser-py",
        "osqp",
        "cvxopt",
        "xacro",
        "vtk",
        "pyyaml",
    ],
    extras_require={
        "example": ["pybullet", "matplotlib"],
        "test": ["roboticstoolbox-python", "pybullet", "pytest", "matplotlib"],
    },
)
