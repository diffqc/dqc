import os
from setuptools import setup, find_packages

module_name = "ddft"
ext_name = "%s.csrc" % module_name
cpp_sources = [
    "src/coeffs.cpp",
    "src/bind.cpp",
]

def get_pybind_include():
    import pybind11
    return pybind11.get_include()

def get_torch_cpp_extension():
    from torch.utils.cpp_extension import CppExtension
    return CppExtension(
        name=ext_name,
        sources=cpp_sources,
        include_dirs=[get_pybind_include()],
        extra_compile_args=['-g'],
    )

def get_build_extension():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension

setup(
    name='ddft',
    version="0.1.0",
    description='Differentiable Density Functional Theory',
    url='https://github.com/mfkasim91/ddft',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "matplotlib>=1.5.3",
        "pybind11>=2.5.0",
    ],
    setup_requires=["pybind11>=2.5.0"],
    ext_modules=[get_torch_cpp_extension()],
    cmdclass={'build_ext': get_build_extension()},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="project library deep-learning dft",
    zip_safe=False
)
