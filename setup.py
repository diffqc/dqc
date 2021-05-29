import os
import sys
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

module_name = "dqc"
file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "_version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

############## build extensions ##############
ext_name = "_dqc_lib_placeholder"
def get_all_libraries(ext=".so"):
    res = []
    for root, dirs, files in os.walk("dqc"):
        for file in files:
            if ext in file:
                 res.append(os.path.relpath(os.path.join(root, file)))
    return res

class CMakeBuildExt(build_ext):
    def run(self):
        extension = self.extensions[0]
        assert extension.name == ext_name
        self.build_extension(self.extensions[0])

    def build_extension(self, ext):
        try:
            self.construct_extension()
        except:
            import warnings
            msg = "Cannot install the extension. Some features might be missing. "
            msg += "Please fix the bug and rerun it with 'python setup.py build_ext'"
            warnings.warn(msg)

        # copy all the libraries to build_lib
        self.announce(f"Moving the libraries to {self.build_lib}")
        lib_paths = get_all_libraries(ext=".so") + get_all_libraries(ext=".dylib")
        for src_lib_path in lib_paths:
            dst_lib_path = os.path.join(self.build_lib, src_lib_path)
            self.announce(f"Moving from {src_lib_path} to {dst_lib_path}")
            os.makedirs(os.path.dirname(dst_lib_path), exist_ok=True)
            shutil.copyfile(src_lib_path, dst_lib_path)

    def construct_extension(self):
        # libraries from PySCF
        lib_dir = os.path.join(file_dir, "dqc", "lib")
        build_dir = self.build_temp
        self.announce(f'Compiling libraries from PySCF from {lib_dir} to {build_dir}', level=3)
        self.build_cmake(lib_dir, build_dir)

    def build_cmake(self, lib_dir, build_dir):
        self.announce("Configuring cmake", level=3)
        cmd = ['cmake', f'-S{lib_dir}', f'-B{build_dir}']
        self.spawn(cmd)

        self.announce("Building binaries", level=3)
        cmd = ['cmake', '--build', build_dir, '-j']
        self.spawn(cmd)

    def python_install(self, dir, fname, mode):
        curpath = os.getcwd()
        try:
            os.chdir(dir)
            self.announce(f"Installing {fname}")
            cmd = [sys.executable, fname, mode]
            self.spawn(cmd)
        finally:
            os.chdir(curpath)

setup(
    name=module_name,
    version=version["get_version"](),
    description='Differentiable Quantum Chemistry',
    url='https://github.com/diffqc/dqc/',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='Apache License 2.0',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "matplotlib>=1.5.3",
        "basis_set_exchange",
        "h5py>=3.1.0",
        "pylibxc2>=6.0.0",
        "xitorch >= 0.3",
        "torch>=1.8",  # ideally the nightly build
    ],
    ext_modules=[Extension(ext_name, [])],
    cmdclass={'build_ext': CMakeBuildExt},
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
