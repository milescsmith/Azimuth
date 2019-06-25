#!/usr/bin/env python

from setuptools import setup

setup(
    name="Azimuth",
    version="3.1",
    author="Nicolo Fusi, Jennifer Listgarten, and Miles Smith",
    author_email="fusi@microsoft.com, jennl@microsoft.com, mileschristiansmith@gmail.com",
    description="Machine Learning-Based Predictive Modelling of CRISPR/Cas9 guide efficiency",
    url="https://github.com/milescsmith/azimuth",
    license="BSD3",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD-3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["azimuth", "azimuth.features", "azimuth.models", "azimuth.tests"],
    keywords="CRISPR",
    project_urls={"Forked_from": "https://github.com/MicrosoftResearch/Azimuth"},
    python_requires=">=3.6",
    package_dir={"azimuth": "azimuth"},
    package_data={"azimuth": ["saved_models/*.*", "data/*.*"]},
    install_requires=[
        "click",
        "biopython",
        "scipy",
        "numpy",
        "scikit-learn",
        "pandas",
        "GPy",
        "hyperopt",
        "paramz",
        "theanets @ git+https://github.com/lmjohns3/theanets",
        "glmnet_py",
        "dill",
    ],
)
