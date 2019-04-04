from setuptools import setup, find_packages

setup(name='py_rfq_helper',
      version='0.0.1',
      description='A set of helper functions to simulate an RFQ in WARP',
      url='https://github.com/DanielWinklehner/py_rfq_helper',
      author='Jared Hwang, Daniel Winklehner',
      author_email='winklehn@mit.edu',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
