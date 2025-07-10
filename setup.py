# Licensed under BSD-3-Clause License - see LICENSE

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

version = {}
with open("StarStream/version.py") as fp:
    exec(fp.read(), version)

setup(
    name = "StarStream",
    packages = find_packages(where="StarStream"),
    version = version["__version__"],
    url = "https://github.com/ybillchen/StarStream",
    license = "BSD-3-Clause",
    author = "Bill Chen",
    author_email = "ybchen@umich.edu",
    description = "An automatic detection algorithm for stellar streams.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    install_requires = ["numpy", "scipy", "astropy", "agama"],
    python_requires = ">=3.9",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)