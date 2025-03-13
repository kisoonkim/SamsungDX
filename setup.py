from setuptools import setup, find_packages
import time
import os

setup(name='NSCMCommon',
      packages=['NSCMCommon'],
      include_package_data=True,
      package_data = {'NSCMCommon' : ['*.csv']},
      zip_safe = False
      )
