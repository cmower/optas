from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyinvk',
    version='1.0.1',
    description='Python Inverse Kinematics.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/cmower/pyinvk',
    project_urls={
        "Bug Tracker": "https://github.com/cmower/pyinvk/issues",
    },
    author='Christopher E. Mower',
    author_email='cmower@ed.ac.uk',
    license='BSD 2-Clause License',
    packages=['pyinvk'],
    install_requires=[
        'numpy',
        'casadi',
        'urdf2casadi',
        'scipy',
    ]
)
