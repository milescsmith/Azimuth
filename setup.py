#!/usr/bin/env python
import sys

if sys.version_info < (3,):
    sys.exit("Azimuth requires Python >= 3.6")
from pathlib import Path
from setuptools import setup, find_packages

try:
    from azimuth import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ""

setup(
    name="Azimuth",
    version="3.3",
    description="Machine Learning-Based Predictive Modelling of CRISPR/Cas9 guide efficiency",
    long_description=Path("README.md").read_text("utf-8"),
    url="https://github.com/milescsmith/azimuth",
    author=__author__,
    author_email=__email__,
    license="BSD3",
    python_requires=">=3.6",
    install_requires=[
        l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD-3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    keywords="CRISPR",
    package_dir={"azimuth": "azimuth"},
    package_data={"azimuth": ["saved_models/*.*", "data/*.*"]},
    include_package_data=True,
    project_urls={"Forked_from": "https://github.com/MicrosoftResearch/Azimuth"},
    entry_points={"console_scripts": ["azimuth = azimuth.__main__:main"]},
)
