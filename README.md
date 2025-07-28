# StarStream

[![version](https://img.shields.io/badge/version-0.1-blue.svg)](https://github.com/ybillchen/StarStream)
[![license](https://img.shields.io/github/license/ybillchen/StarStream)](LICENSE)
[![ADS](https://img.shields.io/badge/ADS-2025xxxxx-blue)](#)
[![arXiv](https://img.shields.io/badge/arXiv-25xx.xxxxx-green)](#)

An automatic detection algorithm for stellar streams using a physics-inspired stream model.

The code is open source under a [BSD 3-Clause License](LICENSE), which allows you to redistribute and modify the code with moderate limitations. If you use this code for a publication, we kindly request you to cite the following original paper.

- Y. Chen, O. Y. Gnedin, A. M. Price-Whelan, & C. Holm-Hansen (2025) *StarStream: Automatic detection algorithm for stellar streams*. [arXiv:25xx.xxxxx](https://arxiv.org/abs/25xx.xxxxx), [ADS link](#)

## Install

We have tested `StarStream` on `3.9 <= python <= 3.11`. However, lower or higher versions may also work. The prerequisites of this package are
```
numpy
scipy
astropy
agama
```

To download the packge, `git clone` the source code from [GitHub](https://github.com/ybillchen/StarStream):
```shell
$ git clone https://github.com/ybillchen/StarStream.git
```
Next, `cd` the folder and use `pip` to install it:
```shell
$ cd StarStream/
$ pip install -e .
```
The `-e` command allows you to make changes to the code.

To check if the package is installed correctly, you may run the tests using `pytest` (make sure it's installed)
```shell
$ pytest
```

## Usage

We provide [example notebooks](examples/) to demonstrate how to use apply this method to a mock dataset similar to *Gaia* DR3.


## Contribute

Feel free to dive in! [Raise an issue](https://github.com/ybillchen/StarStream/issues/new) or submit pull requests. We recommend you to contribute code following [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow). 

## Maintainers

- [@ybillchen (Bill Chen)](https://github.com/ybillchen)

