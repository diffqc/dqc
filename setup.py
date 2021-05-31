import os
import sys
import subprocess as sp
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# module's descriptor
module_name = "dqc"
ext_name = "dqc.lib.pyscflibs"
github_url = "https://github.com/diffqc/dqc/tree/master/"
raw_github_url = "https://raw.githubusercontent.com/diffqc/dqc/master/"

file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

# get the long description from README
# open readme and convert all relative path to absolute path
with open("README.md", "r") as f:
    long_desc = f.read()

link_pattern = re.compile(r"\(([\w\-/]+)\)")
img_pattern  = re.compile(r"\(([\w\-/\.]+)\)")
link_repl = r"(%s\1)" % github_url
img_repl  = r"(%s\1)" % raw_github_url
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

############## build extensions ##############
def get_all_libraries(ext=".so"):
    res = []
    for root, dirs, files in os.walk("dqc"):
        for file in files:
            if ext in file:
                 res.append(os.path.relpath(os.path.join(root, file)))
    return res

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuildExt(build_ext):
    def run(self):
        extension = self.extensions[0]
        assert extension.name == ext_name
        self.build_extension(self.extensions[0])

    def build_extension(self, ext):
        if "NO_DQC_EXT_NEEDED" in os.environ:
            try:
                self.construct_extension(ext)
            except:
                import warnings
                msg = "Cannot install the extension. Some features might be missing. "
                msg += "Please fix the bug and rerun it with 'python setup.py build_ext'"
                warnings.warn(msg)
        else:
            self.construct_extension(ext)

    def construct_extension(self, ext):
        # libraries from PySCF
        lib_dir = os.path.join(file_dir, "dqc", "lib")
        build_dir = self.build_temp
        self.announce(f'Compiling libraries from PySCF from {lib_dir} to {build_dir}', level=3)
        self.build_cmake(ext, lib_dir, build_dir)

    def build_cmake(self, ext, lib_dir, build_dir):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        self.announce("Configuring cmake", level=3)
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]
        cmd = ['cmake', f'-S{lib_dir}', f'-B{build_dir}'] + cmake_args
        self.spawn(cmd)

        self.announce("Building binaries", level=3)
        cmd = ['cmake', '--build', build_dir, '-j']
        self.spawn(cmd)

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
        "xitorch >= 0.3",
        "torch>=1.8",  # ideally the nightly build
        "cmake>=3.0.0",
    ],
    ext_modules=[CMakeExtension(ext_name, '')],
    cmdclass={'build_ext': CMakeBuildExt},
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
