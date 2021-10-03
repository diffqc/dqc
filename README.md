# DQC: Differentiable Quantum Chemistry

![Build](https://img.shields.io/github/workflow/status/diffqc/dqc/ci?style=flat-square)
[![Code coverage](https://img.shields.io/codecov/c/github/diffqc/dqc?style=flat-square)](https://app.codecov.io/gh/diffqc/dqc)
[![Docs](https://img.shields.io/readthedocs/dqc?style=flat-square)](https://dqc.readthedocs.io/)

Differentiable quantum chemistry package.
Currently only support differentiable density functional theory (DFT)
and Hartree-Fock (HF) calculation.

Installation, tutorials, and documentations can be found at: https://dqc.readthedocs.io/

## Applications

Here is a list of applications made easy by DQC.
If you want your applications listed here, please contact us by opening an issue
or make a pull request.

<!-- start of readme_appgen.py -->
<!-- Please do not edit this part directly, instead add your application in the readme_appgen.py file -->
| Applications                      | Repo | Paper |
|-----------------------------------|------|-------|
| Learning xc functional from experimental data | [![github](docs/data/readme_icons/github.svg)](https://github.com/mfkasim1/xcnn) | [![Paper](docs/data/readme_icons/paper.svg)](https://arxiv.org/abs/2102.04229) |
| Basis optimization | [![github](docs/data/readme_icons/github.svg)](https://github.com/diffqc/dqc-apps/tree/main/01-basis-opt) |  |
| Alchemical perturbation | [![github](docs/data/readme_icons/github.svg)](https://github.com/diffqc/dqc-apps/tree/main/04-alchemical-perturbation) |  |
<!-- end of readme_appgen.py -->

## Citations

If you are using DQC for your publication, please kindly cite the following

    @article{PhysRevLett.127.126403,
      title = {Learning the Exchange-Correlation Functional from Nature with Fully Differentiable Density Functional Theory},
      author = {Kasim, M. F. and Vinko, S. M.},
      journal = {Phys. Rev. Lett.},
      volume = {127},
      issue = {12},
      pages = {126403},
      numpages = {7},
      year = {2021},
      month = {Sep},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevLett.127.126403},
      url = {https://link.aps.org/doi/10.1103/PhysRevLett.127.126403}
    }

If you want to read the paper in arxiv, you can find it [here](https://arxiv.org/abs/2102.04229).
