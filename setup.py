#!/usr/bin/env python

from setuptools import setup
# import os
#
# os.system('pip install git+https://github-private.corp.com/user/repo.git@master')

setup(name='Azimuth',
      version='2.5',
      author='Nicolo Fusi, Jennifer Listgarten, and Miles Smith',
      author_email="fusi@microsoft.com, jennl@microsoft.com, mileschristiansmith@gmail.com",
      description="Machine Learning-Based Predictive Modelling of CRISPR/Cas9 guide efficiency",
      url='https://github.com/milescsmith/azimuth',
      license='BSD3',
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: BSD-3',
                   'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.7'],
      packages=["azimuth", "azimuth.features", "azimuth.models", "azimuth.tests"],
      keywords='CRISPR',
      project_urls={
            'Forked_from': 'https://github.com/MicrosoftResearch/Azimuth'
      },
      python_requires='>=3.6',
      package_dir={
            'azimuth': 'azimuth'
      },
      package_data={
            'azimuth': ['saved_models/*.*','data/*.*']
      },
      install_requires=['scipy', 'numpy', 'nose', 'scikit-learn', 'pandas', 'biopython','GPy','ipyparallel', 'mkl',
                        'hyperopt', 'paramz', 'theanets==0.8.0rc0', 'glmnet_py', 'xlrd'],
      dependency_links=['https://github.com/lmjohns3/theanets/tarball/master#egg=theanets-0.8.0rc0']
      )
