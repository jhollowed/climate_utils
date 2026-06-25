# This is the main setup config file, which allows local installaton of the package via pip.
# From the top-level directory (location of this file), simply run `pip install .`
# For more info, see the guidelines for minimal Python package structure at
# https://python-packaging.readthedocs.io/en/latest/minimal.html

from setuptools import setup, find_packages

setup(name='climate_utils',
      version='0.1',
      description='Package providing convenience functions for analysis and plotting of climate modeling datasets',
      url='https://github.com/jhollowed/climate_utils',
      author='Joe Hollowed',
      author_email='hollowed@umich.edu',
      packages=find_packages(),
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
