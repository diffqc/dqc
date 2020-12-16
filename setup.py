import os
from setuptools import setup, find_packages

setup(
    name='ddft',
    version="0.1.0",
    description='Differentiable Density Functional Theory',
    url='https://github.com/mfkasim1/ddft',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "matplotlib>=1.5.3",
        "basis_set_exchange",
        # "pytorch>=1.8.0",
    ],
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
