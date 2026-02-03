#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages
from setuptools import setup


with open('README.rst', 'rt') as readme_file:
    readme = readme_file.read()


def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv('CIRCLE_BRANCH') in {'master'}:
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='multic',
    use_scm_version={'local_scheme': prerelease_local_scheme},
    description='multi comprtment segmentation, feature extraction',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Anish Tatke',
    author_email='anish.tatke@ufl.edu',
    url='https://github.com/SarderLab/Multi-Compartment-Segmentation',
    packages=find_packages(exclude=['tests', '*_test']),
    package_dir={
        'multic': 'multic',
    },
    include_package_data=True,
    install_requires=[
        # scientific packages
        "nimfa>=1.4.0",
        "numpy>=1.26.4",
        "scipy>=1.11.4",
        "Pillow>=10.2.0",
        "pandas>=2.2.0",
        "imageio>=2.34.0",
        "pyvips>=2.2.2",
        "termcolor>=2.4.0",
        "opencv-python>=4.9.0.80",
        "scikit-image>=0.22.0",
        "lxml>=5.1.0",
        "joblib>=1.3.2",
        "tiffslide>=2.0.0",
        "tqdm>=4.66.0",
        "openpyxl>=3.1.2",
        "xlrd<2",
        "dask[dataframe]>=2024.1.0",
        "distributed>=2024.1.0",
        # girder
        'girder-slicer-cli-web',
        'girder-client',
        # cli
        'ctk-cli',
    ],
    license='Apache Software License 2.0',
    keywords='multic',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)
