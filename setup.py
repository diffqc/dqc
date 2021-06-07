import os
import sys
import re
import subprocess as sp
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# module's descriptor
module_name = "dqc"
github_url = "https://github.com/diffqc/dqc/tree/master/"
raw_github_url = "https://raw.githubusercontent.com/diffqc/dqc/master/"

file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

# get the long description from README
# open readme and convert all relative path to absolute path
with open("README.md", "r") as f:
    long_desc = f.read()

link_pattern = re.compile(r"\]\(([\w\-/]+)\)")
img_pattern  = re.compile(r"\]\(([\w\-/\.]+)\)")
link_repl = r"](%s\1)" % github_url
img_repl  = r"](%s\1)" % raw_github_url
long_desc = re.sub(link_pattern, link_repl, long_desc)
long_desc = re.sub(img_pattern, img_repl, long_desc)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "_version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

# execute _version.py to create _version.txt
cmd = [sys.executable, verfile]
sp.run(cmd)

vers = version["get_version"]()
setup(
    name=module_name,
    version=vers,
    description='Differentiable Quantum Chemistry',
    url='https://github.com/diffqc/dqc/',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='Apache License 2.0',
    packages=find_packages(),
    package_data={module_name: ["_version.txt", "datasets/lebedevquad/lebedev_*.txt"]},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "basis_set_exchange",
        "h5py>=3.1.0",
        "pylibxc2>=6.0.0",
        "dqclibs>=0.1.0",
        "xitorch>=0.3",
        "torch>=1.8",  # ideally the nightly build
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",

        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="project library deep-learning dft quantum-chemistry",
    zip_safe=False
)
