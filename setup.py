import os
from setuptools import setup, find_packages

module_name = "dqc"
file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

setup(
    name=module_name,
    version=version["get_version"](),
    description='Differentiable Quantum Chemistry',
    url='https://github.com/mfkasim1/dqc/',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='Apache License 2.0',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "matplotlib>=1.5.3",
        "basis_set_exchange",
        "h5py>=3.1.0",
        "xitorch @ git+https://github.com/xitorch/xitorch.git",
        "torch>=1.7.1",  # ideally the nightly build (1.8.0), but we will just have 1.7.1 for the moment
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",

        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="project library deep-learning dft quantum-chemistry",
    zip_safe=False
)
