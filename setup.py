import os
from setuptools import setup, find_packages

setup(
    name='deepemulator',
    version="0.1.0",
    description='Deep emulator networks project library',
    url='https://github.com/mfkasim91/deepemulator',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=2.7",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "matplotlib>=1.5.3",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 2.7"
    ],
    keywords="project library deep-learning",
    zip_safe=False
)
